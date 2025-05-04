# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import act_layer, DyGraphConv2d, get_2d_relative_pos_embed

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import print_log
from .pyramid_vig import Stem, Downsample, default_cfgs
from .pyramid_vig_vp import PadPrompter, FixedPatchPrompter, RandomPatchPrompter
import sklearn.cluster as cluster
from tqdm import tqdm
class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=nn.LayerNorm,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)
        
    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm, bias, 
                                              stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx], relative_pos=False),
                                      FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx]))]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.downstream_head = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                                   nn.BatchNorm2d(1024),
                                   act_layer(act),
                                   nn.Dropout(opt.dropout),
                                   nn.Conv2d(1024, opt.n_classes, 1, bias=True)
                                   )
        self.prototype_gather = None
        self.prompter_gather = None
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    # Diversity-Aware Meta Visual Prompting modules
    def forward_features(self, inputs):
        inputs = inputs
        x = self.stem(inputs) + self.pos_embed
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1)
        # x = self.downstream_head(x)
        x = x.squeeze(-1).squeeze(-1)
        return x                
    
    def coarse_clustering(self, train_loader):
        """Diversity-Aware Adaption on downstream data.
        We roughly divide the downstream task data into several partitions, each 
        partition presents a coarsely divided cluster. Different clusters correspond 
        to different prompters. 
        """
        threshold_dict = {
            "resnet50-1k": 21, 
            "vit-b-1k": 31, 
            "vit-b-22k": 10, 
            "swin-b-22k": 20, 
            "moco-v3-b-1k":18
        }
        hc = cluster.AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=35)
        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
                image = image.cuda()
                rep = self.forward_features(image)
                rep_gather = rep if batch_idx < 1 else torch.cat([rep_gather, rep], dim=0)
                if rep_gather.size(0) > 1000:
                    rep_gather = rep_gather[:1000]
                    break
        y_pred = hc.fit(rep_gather.detach().cpu().numpy()).labels_
        y_pred = torch.from_numpy(y_pred).cuda()
        coarse_class_idx = torch.unique(y_pred)
        self.num_coarse_classes = len(coarse_class_idx)
        print("Nums of coarsely divided categories for test dataset: {}".format(len(coarse_class_idx)))
        prototype_gather = []
        for i in range(len(coarse_class_idx)):
            pos = torch.where(y_pred == i)[0]
            prototype = rep_gather[pos].mean(0).unsqueeze(0)
            prototype_gather.append(prototype)
        self.prototype_gather = torch.cat(prototype_gather)
        self.prompter_gather = nn.ModuleList([PadPrompter(224, 32) for _ in range(len(coarse_class_idx))])
        print("Nums of prototypes of coarse clusters for test dataset: {}".format(self.prototype_gather.size(0)))

    def get_prompted_image(self, image):
        """Obtain the prompted batch images."""
        assert self.prototype_gather is not None
        assert self.prompter_gather is not None
        prompted_image = []
        with torch.no_grad():
            rep_batch = self.forward_features(image) # [N, emd_dim]
            rep_batch_sum = (rep_batch**2).sum(dim=-1, keepdims=True) # [N, 1]
            prototype_gather_sum = (self.prototype_gather**2).sum(dim=-1, keepdims=True).T # [1, M]
            distance_matrix = torch.sqrt(rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, self.prototype_gather.T)) # [N, M]
            indices = torch.argmin(distance_matrix, dim=-1) # [B]
        for idx in range(rep_batch.size(0)):
            prompted_image.append(self.prompter_gather[indices[idx]](image[idx].unsqueeze(0)))
        prompted_image = torch.cat(prompted_image, dim=0)
        return prompted_image


    def forward(self, inputs):
        inputs = self.get_prompted_image(inputs)
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.downstream_head(x)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def load_model_from_ckpt(self, ckpt_path, logger=None):
        ckpt = torch.load(ckpt_path)
        incompatible = self.load_state_dict(ckpt, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger=logger)
            print_log(get_missing_parameters_message(incompatible.missing_keys), logger=logger)
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger=logger)
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger=logger)

        print(f'[Transformer] Successful Loading the ckpt from {ckpt_path}')


@register_model
def pvig_damvp_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [48, 96, 240, 384] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_damvp_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [80, 160, 400, 640] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_damvp_m_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,16,2] # number of basic blocks in the backbone
            self.channels = [96, 192, 384, 768] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_damvp_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,18,2] # number of basic blocks in the backbone
            self.channels = [128, 256, 512, 1024] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model
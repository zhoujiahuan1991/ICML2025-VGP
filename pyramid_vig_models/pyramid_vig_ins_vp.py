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
from types import SimpleNamespace

class TokenPrompt(nn.Module):
    def __init__(self, args, in_channels):
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        self.kernel_size_1 = args.TP_kernel_1
        k = self.kernel_size_1
        if self.args.token_prompt_type == "add":
            self.p = args.p_len
        elif self.args.token_prompt_type == "token":
            self.p = args.p_len // 2
        p = self.p
        self.conv1 = nn.Conv2d(3, p, kernel_size=k, padding=int((k-1)/2))
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(4, 4)
        self.dropout1 = nn.Dropout(0.1)
        self.kernel_size_2 = args.TP_kernel_2
        k = self.kernel_size_2
        self.conv2 = nn.Conv2d(p, 3*p, kernel_size=k, padding=int((k-1)/2))
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(3, 3)
        self.dropout2 = nn.Dropout(0.1)
        self.kernel_size_3 = args.TP_kernel_3
        k = self.kernel_size_3
        self.conv3 = nn.Conv2d(3*p, 3*p, kernel_size=k, padding=int((k-1)/2))
        self.head = nn.Conv2d(42, int(42*self.in_channels/768),1,1)
        self.norm = nn.LayerNorm(self.in_channels)
             
    def forward(self, x):
        x = self.conv1(x) 
        x = self.relu1(x)
        x = x[:,:,8:216,8:216]
        x = self.pool1(x) # [B, 9, 56, 56]
        x = self.dropout1(x)
        x = self.conv2(x) 
        x = self.relu2(x)
        x = x[:,:,2:50,2:50]
        x = self.pool2(x) # [B, 27, 16, 16]
        x = self.dropout2(x)
        x = self.conv3(x) 
        x = self.head(x)
        x = x.reshape(-1, self.p, self.in_channels)
        x = self.norm(x)
        return x

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
        self.node_prompts = nn.Parameter(torch.zeros([in_channels, 4, int(n**0.5)]))
        nn.init.kaiming_normal_(self.node_prompts)
        self.prompt_dropout = nn.Dropout(0.1)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x, instance_prompts):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        node_prompts = self.node_prompts[None,:,:,:]
        node_prompts = self.prompt_dropout(node_prompts.expand([B,-1,-1,-1]))
        if instance_prompts.shape[1] == node_prompts.shape[1]:
            node_prompts = node_prompts + instance_prompts.transpose(-1,-2)
        x = torch.cat([x, node_prompts], dim=-2)
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = x[:,:,:H,:]
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

ins_vp_args = {
    "meta_net": 19,
    "prompt_patch": 16,
    "prompt_patch_12": 31,
    "prompt_patch_2": 11,
    "prompt_patch_22": 25,
    "prompt_patch_3": 64,
    "hid_dim": 16,
    "hid_dim_2": 6,
    "global_prompts_weight": 1,
    "prompts_2_weight": 1,
    "prompts_3_weight": 1,
    "p_len_vpt": 14,
    "p_len": 14,
    "deep_prompt_type": "ours9",
    "instance_prompt": "instance_prompt",
    "prompt_dropout": 0.1,
    "meta_bn": "none",
    "TP_kernel_1": 3,
    "TP_kernel_2": 3,
    "TP_kernel_3": 3,
    "token_prompt_eta": 0.5,
    "token_prompt_type": "add",
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
        
        # modules of Ins VP
        self.meta_dropout = torch.nn.Dropout(0.1)
        self.meta_dropout_2 = torch.nn.Dropout(0.1)
        self.args = opt.args
        self.prompt_patch = self.args.prompt_patch
        n = self.prompt_patch
        h = self.args.hid_dim
        self.meta_net = nn.Sequential(
            nn.Linear(3*n*n, h),
            nn.ReLU(),
            nn.Linear(h, 3*n*n)
        )   
        self.prompt_patch_2 = self.args.prompt_patch_2
        self.prompt_patch_22 = self.args.prompt_patch_22
        n_2 = self.prompt_patch_2
        n_22 = self.prompt_patch_22
        self.meta_net_2 = nn.Sequential(
            nn.Conv2d(3, self.args.hid_dim_2, n_2, stride=1, padding=int((n_2-1)/2)),
            nn.ReLU(),
            nn.Conv2d(self.args.hid_dim_2, 3, n_22, stride=1, padding=int((n_22-1)/2))
        ) 
        self.InsTokenPrompt = TokenPrompt(self.args, channels[-2])

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
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    # modules for InsVP
    def get_local_prompts(self, x):
        # [64, 3, 224, 224]
        B = x.shape[0]
        n = self.prompt_patch
        n_patch = int(224 / n)
        x = x.reshape(B, 3, n_patch, n, n_patch, n) # [64, 3, 14, 16, 14, 16]
        x = x.permute(0, 2, 4, 1, 3, 5) # [64, 14, 14, 3, 16, 16]
        x = x.reshape(B, n_patch*n_patch, 3*n*n)
        x = x.reshape(B*n_patch*n_patch, 3*n*n)
        x = self.meta_net(x)
        x = x.reshape(B, n_patch, n_patch, 3, n, n)
        x = x.permute(0, 3, 1, 4, 2, 5) # [64, 3, 14, 16, 14, 16]
        x = x.reshape(B, 3, 224, 224)
        return self.meta_dropout(x)

    def get_prompts(self, x):
        prompts_1 = self.get_local_prompts(x)
        x = self.meta_dropout_2(self.meta_net_2(x))
        return prompts_1 + self.args.prompts_2_weight * x

    def forward(self, inputs):
        original_image = inputs
        prompts = self.get_prompts(inputs)  
        inputs = inputs + prompts*0.02
        x = self.stem(inputs)
        x = x + self.pos_embed
        B, C, H, W = x.shape
        instance_prompts = self.InsTokenPrompt(original_image)[..., None].transpose(1,2)
        for i in range(len(self.backbone)):
            if type(self.backbone[i]) is Downsample:
                x = self.backbone[i](x)
            else:
                x = self.backbone[i][0](x, instance_prompts)
                x = self.backbone[i][1](x)

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
def pvig_ins_vp_ti_224_gelu(pretrained=False, **kwargs):
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
            self.args = SimpleNamespace(**ins_vp_args)

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_ins_vp_s_224_gelu(pretrained=False, **kwargs):
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
            self.args = SimpleNamespace(**ins_vp_args)

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_ins_vp_m_224_gelu(pretrained=False, **kwargs):
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
            self.args = SimpleNamespace(**ins_vp_args)

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_ins_vp_b_224_gelu(pretrained=False, **kwargs):
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
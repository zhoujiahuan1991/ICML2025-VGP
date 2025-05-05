import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import warnings
warnings.filterwarnings('ignore')
import time
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.data import resolve_data_config, Mixup
from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from data_utils import loader_vig as data_loader
from utils.logger import *
import pyramid_vig_models
import vig_models
from tqdm import tqdm
from utils.misc import summary_parameters, peft_detect
from data_utils import create_loader, create_dataset
import shutil
from parse import parse_args

has_native_amp = False
head_dim_dict={'cifar100':100, 'cifar10':10, 'dtd47':47, 'food101':101, 'cub200':200, 'stanford_dogs120':120, 'nabirds1011':1011, 'flowers102':102, 'gtsrb43':43, 'svhn10':10,
               'vtab-caltech101': 102, 'vtab-clevr_count': 8, 'vtab-diabetic_retinopathy': 5, 'vtab-dsprites_loc': 16, 'vtab-dtd': 47, 'vtab-kitti': 4, 'vtab-oxford_iiit_pet': 37,
               'vtab-resisc45': 45, 'vtab-smallnorb_ele': 9, 'vtab-svhn': 10, 'vtab-cifar': 100, 'vtab-clevr_dist': 6, 'vtab-dmlab': 6, 'vtab-dsprites_ori': 16, 'vtab-eurosat': 10, 'vtab-oxford_flowers102': 102, 'vtab-patch_camelyon': 2, 'vtab-smallnorb_azi': 18, 'vtab-sun397': 397 
               }
torch.backends.cudnn.benchmark = True

def main():
    setup_default_logging()
    args, args_text = parse_args()
    args.num_classes = head_dim_dict[args.dataset]

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            print_log('Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        print_log('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'%(args.rank, args.world_size))
    else:
        print_log('Training with a single process on %d GPUs.'%args.num_gpu)

    torch.manual_seed(args.seed + args.rank)
    
    relative_pos = False
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        relative_pos = relative_pos
        )
    
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    output_dir = ''
    if args.local_rank == 0:
        output_base = './experiments'
        exp_tag = '-'.join([datetime.now().strftime(f"%Y%m%d-%H%M%S"), str(data_config['input_size'][-1])])
        output_dir = get_outdir(output_base, args.dataset.replace('-','/'), args.model, args.exp_name, exp_tag)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        log_file = os.path.join(output_dir, f'history.log')
        logger = get_root_logger(log_file=log_file, name=args.exp_name)

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, 
        root=args.data_dir, 
        split='train', 
        is_training=True,
        batch_size=args.batch_size
        )

    dataset_eval = create_dataset(
        args.dataset, 
        root=args.data_dir, 
        split='test', 
        is_training=False,
        batch_size=args.batch_size
        )

    # setup mixup / cutmix
    mixup_fn = None
    if args.use_mixup:
        # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        mixup_args = dict(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                          prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                          label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    train_loader = create_loader(
        dataset_train,
        input_size=data_config['input_size'][-1],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        num_workers=args.workers,
        distributed=args.distributed,
        pin_memory=args.pin_mem,
        args=args
    )

    eval_loader = create_loader(
        dataset_eval,
        input_size=data_config['input_size'][-1],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        num_workers=args.workers,
        distributed=args.distributed,
        pin_memory=args.pin_mem,
        args=args
    )

    # pretrain
    if args.pretrain_path is not None:
        model.load_model_from_ckpt(args.pretrain_path)
        print_log('Pretrain weights loaded.', logger=logger)
    

    model.cuda()

    print_log("Require gradient parameters: ", logger = logger)
    if 'vpt' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'node_prompts']
    elif 'vig_vp_' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'visual_prompts']
    elif 'adapter' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'adapter']
    elif 'ins_vp' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'node_prompts', 'meta_net', 'meta_net_2', 'InsTokenPrompt']
    elif 'graph_prompt' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'node_prompts', 
                     'node_prompter', 'adapter', 'edge_prompter', 'low_rank_edge_prompts']
    elif 'lor_gp' in args.model:
        peft_list = ['downstream_head', 'pos_embed','node_prompts', 'graph_prompt', 
                     'node_prompter', 'group_prompt', 'edge_prompt', 'node_prompt'] 
    elif 'vfpt' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'node_prompts', 'FT']
    elif 'damvp' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'prompter_gather']
    elif 'gpf' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'gpf_prompt']
    elif 'gprompt' in args.model:
        peft_list = ['downstream_head', 'pos_embed', 'gprompt']
    else:
        peft_list = ['downstream_head', 'pos_embed'] # linear probing
    
    if 'damvp_' in args.model:
        model.coarse_clustering(train_loader)
        model.cuda()
    
    if args.peft:
        for name, param in model.named_parameters():
            if peft_detect(name, peft_list): 
                print_log(name, logger = logger)
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    summary_parameters(model, logger=logger) #logger=logger

    from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, (3,224,224), as_strings=True)
    # print_log(f"FLOPs: {flops}, Params: {params}", logger=logger)
    # # with open('efficiency1.txt',"a+") as f:
    #     # f.write(f"{args.dataset}: FLOPs: {flops}, Params: {params} \n")
    # with open('efficiency2.txt',"a+") as f:
    #     f.write(f"{args.dataset}: FLOPs: {flops}, Params: {params} \n")
    # return None

    if args.distributed:
        if args.local_rank == 0:
            print_log("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.local_rank == 0:
        print_log('Scheduled epochs: {}'.format(args.epochs), logger=logger)
        print_log('Cool down epochs: {}'.format(args.cooldown_epochs), logger=logger)

    optimizer = create_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            from timm.models import load_checkpoint
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    saver = None
    eval_metric = args.eval_metric
    if args.local_rank == 0:
        print_log('Model %s created, param count: %d'%(args.model, sum([m.numel() for m in model.parameters()])), logger=logger)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model, optimizer=optimizer, args=args, model_ema=model_ema, checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=1)

    if args.model.startswith('pvig'):
        shutil.copy('pyramid_vig_models/%s.py' % args.model.split('_b')[0].split('_m')[0].replace('pvig','pyramid_vig'), str(output_dir))
    elif args.model.startswith('vig'):
        shutil.copy('vig_models/%s.py' % args.model.split('_b')[0].split('_m')[0], str(output_dir))
    else:
        raise NotImplementedError
    shutil.copy('main.py', str(output_dir))
    shutil.copy('data_utils/transforms.py', str(output_dir))
    shutil.copy('parse.py', str(output_dir))
    if not 'vtab' in args.dataset:
        print_log("Nums of classes: {}".format(len(dataset_train.classes)), logger=logger)
        print_log("Sample nums: train-{}, val-{}".format(len(dataset_train), len(dataset_eval)), logger=logger)
        assert args.num_classes==len(dataset_train.classes), "The number of classes of prediction head has conflicts to that of dataset."
    
        
    if args.smoothing:
        train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    best_metric = 0
    best_epoch = -1

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print_log(f"Epoch {epoch} start:", logger=logger)
            train_metrics = train_epoch(epoch, model, train_loader, optimizer, train_loss_fn, args,
                                        lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                                        model_ema=model_ema, mixup_fn=mixup_fn, logger=logger)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    print_log("Distributing BatchNorm running means and vars", logger=logger)
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, eval_loader, validate_loss_fn, args, logger=logger)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(model_ema.ema, eval_loader, validate_loss_fn, args, log_suffix='[EMA]', logger=logger)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'), write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                print_log('train_metrics: '+str(train_metrics), logger=logger)
                print_log('eval_metrics: '+str(eval_metrics), logger=logger)
                if save_metric > best_metric:
                    print_log(f"---------------------------------------------Best-Result-{save_metric}--------------------------------------", logger=logger)
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        print_log('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch), logger=logger)


def train_epoch(epoch, model, loader, optimizer, loss_fn, args, lr_scheduler=None, saver=None, output_dir='', model_ema=None, mixup_fn=None, logger=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(tqdm(loader)):
        data_time_m.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
    
        output = model(input)
        loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss.backward(create_graph=second_order)
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if batch_idx == last_idx or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
            if args.local_rank == 0:
                print_log(
                    'Dataset: {} '
                    'Train {} epochs: [{:d}/{}({:.0f}%)] '
                    'Loss:{loss.val:.3f}({loss.avg:.2f}) '
                    'LR:{lr} '.format(
                        args.dataset,
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr),
                        logger=logger if batch_idx == last_idx else None
                        )
                if args.save_images and output_dir:
                    torchvision.utils.save_image(input, os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx), padding=0, normalize=True)

        if saver is not None and args.recovery_interval and (batch_idx == last_idx or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, log_suffix='', logger=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            input = input.cuda()
            target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, min(args.num_classes, 5)))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (batch_idx == last_idx or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                print_log('{0}:[{1:d}/{2}] '
                        'Dataset: {dataset} '
                        'Loss:{loss.val:.4f}({loss.avg:.4f}) '
                        'Acc@1:{top1.val:.4f}({top1.avg:.4f}) '
                        'Acc@5:{top5.val:.4f}({top5.avg:.4f}) '.format(
                        log_name, batch_idx, last_idx, dataset=args.dataset,
                        loss=losses_m, top1=top1_m, top5=top5_m), logger=logger)
                
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics

if __name__ == '__main__':
    main()

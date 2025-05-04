import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
import vig_graph_prompt
from tqdm import tqdm
from utils.misc import summary_parameters, peft_detect
from data_utils import create_loader, create_dataset
import shutil
from parse import parse_args
from main import head_dim_dict, validate

torch.backends.cudnn.benchmark = True

def test():
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
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path
        )
    
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    # output_dir = ''
    # if args.local_rank == 0:
    #     output_base = './experiments'
    #     exp_tag = '-'.join([datetime.now().strftime(f"%Y%m%d-%H%M%S"), str(data_config['input_size'][-1])])
    #     output_dir = get_outdir(output_base, args.dataset.replace('-','/'), args.model, args.exp_name, exp_tag)
    #     with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
    #         f.write(args_text)
    #     log_file = os.path.join(output_dir, f'history.log')
    #     logger = get_root_logger(log_file=log_file, name=args.exp_name)

    
    dataset_eval = create_dataset(
        args.dataset, 
        root=args.data_dir, 
        split='test', 
        is_training=False,
        batch_size=args.batch_size
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
    assert args.initial_checkpoint is not None, "The test checkpoint should be given."
    model.load_model_from_ckpt(args.initial_checkpoint)
    print_log('Pretrain weights loaded.', logger=None)

    model.cuda()

    from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, (3,224,224), as_strings=True)
    # print_log(f"FLOPs: {flops}, Params: {params}", logger=logger)

    if args.distributed:
        if args.local_rank == 0:
            print_log("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.local_rank == 0:
        print_log('Scheduled epochs: {}'.format(args.epochs), logger=None)
        print_log('Cool down epochs: {}'.format(args.cooldown_epochs), logger=None)

    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    eval_metrics = validate(model, eval_loader, validate_loss_fn, args, logger=None)
    print(eval_metrics)

    

if __name__ == '__main__':
    test()

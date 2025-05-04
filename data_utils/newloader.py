#!/usr/bin/env python3

"""Data loader."""
import os
import sys
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from data.myloader import create_loader  # 引入 Vision GNN 的数据加载器

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from data_utils.datasets import *

_DATASET_CATALOG = {
    ### preparing for meta training
    "sun397": SUN397,
    "stl10": STL10,
    "fru92": Fru92Dataset,
    "veg200": Veg200Dataset,
    "oxford-iiit-pets": OxfordIIITPet,
    "eurosat": EuroSAT,
    ### preparing for task adapting
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "cub200": CUB200Dataset,
    "nabirds": NabirdsDataset,
    "oxford-flowers": FlowersDataset,
    "stanford-dogs": DogsDataset,
    "stanford-cars": CarsDataset,
    "fgvc-aircraft": AircraftDataset,
    "food101": Food101,
    "dtd": DTD,
    "svhn": SVHN,
    "gtsrb": GTSRB,
    # 添加 Vision GNN 数据集
    "imagenet": ImageNetDataset,
}

_DATA_DIR_CATALOG = {
    ### preparing for meta training
    "sun397": "torchvision_dataset/",
    "stl10": "torchvision_dataset/",
    "fru92": "finegrained_dataset/vegfru-dataset",
    "veg200": "finegrained_dataset/vegfru-dataset",
    "oxford-iiit-pets": "torchvision_dataset/",
    "eurosat": "torchvision_dataset/",
    ### preparing for task adapting
    "cifar10": "torchvision_dataset/",
    "cifar100": "torchvision_dataset/",
    "cub200": "FGVC/CUB_200_2011/",
    "nabirds": "FGVC/nabirds/",
    "oxford-flowers": "FGVC/OxfordFlower/",
    "stanford-dogs": "FGVC/Stanford-dogs/",
    "stanford-cars": "FGVC/Stanford-cars/",
    "fgvc-aircraft": "FGVC/fgvc-aircraft-2013b/",
    "food101": "torchvision_dataset/",
    "dtd": "torchvision_dataset/",
    "svhn": "torchvision_dataset/",
    "gtsrb": "torchvision_dataset/",
    # Vision GNN 数据集目录
    "imagenet": "imagenet_dataset/",
}

_NUM_CLASSES_CATALOG = {
    ### preparing for meta training
    "sun397": 397,
    "stl10": 10,
    "fru92": 92,
    "veg200": 200,
    "oxford-iiit-pets": 37,
    "eurosat": 10,
    ### preparing for task adapting
    "cifar10": 10,
    "cifar100": 100,
    "cub200": 200,
    "nabirds": 555,
    "oxford-flowers": 102,
    "stanford-dogs": 120,
    "stanford-cars": 196,
    "fgvc-aircraft": 100,
    "food101": 101,
    "dtd": 47,
    "svhn": 10,
    "gtsrb": 43,
    # Vision GNN 数据集类别数
    "imagenet": 1000,
}


def get_dataset_classes(dataset):
    """Given a dataset, return the name list of dataset classes."""
    if hasattr(dataset, "classes"):
        return dataset.classes
    elif hasattr(dataset, "_class_ids"):
        return dataset._class_ids
    elif hasattr(dataset, "labels"):
        return dataset.labels
    else:
        raise NotImplementedError


def _construct_loader(args, dataset, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = dataset

    # 构建数据集
    if dataset_name.startswith("vtab-"):
        args.data_dir = os.path.join(args.base_dir, "VTAB/")
        from data_utils.datasets.tf_dataset import TFDataset
        dataset = TFDataset(args, split)
    else:
        assert dataset_name in _DATASET_CATALOG.keys(), "Dataset '{}' not supported".format(dataset_name)
        args.data_dir = os.path.join(args.base_dir, _DATA_DIR_CATALOG[dataset_name])
        dataset = _DATASET_CATALOG[dataset_name](args, split)

    # 使用 Vision GNN 的数据加载器
    loader = create_loader(
        dataset=dataset,
        input_size=args.input_size,
        batch_size=batch_size,
        is_training=(split == 'train'),
        use_prefetcher=args.use_prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=args.num_aug_splits,
        interpolation=interpolation,
        mean=args.mean,
        std=args.std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        collate_fn=args.collate_fn,
        pin_memory=args.pin_memory,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        repeated_aug=args.repeated_aug
    )

    return loader


def construct_train_dataset(args, dataset=None):
    """Train loader wrapper."""
    if args.distributed:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        args=args,
        split="train",
        batch_size=int(args.batch_size / args.num_gpus),
        shuffle=True,
        drop_last=drop_last,
        dataset=dataset if dataset else args.dataset
    )


def construct_val_dataset(args, dataset=None, batch_size=None):
    """Validation loader wrapper."""
    if batch_size is None:
        bs = int(args.batch_size / args.num_gpus)
    else:
        bs = batch_size
    return _construct_loader(
        args=args,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        dataset=dataset if dataset else args.dataset
    )


def construct_test_dataset(args, dataset=None):
    """Test loader wrapper."""
    return _construct_loader(
        args=args,
        split="test",
        batch_size=int(args.batch_size / args.num_gpus),
        shuffle=False,
        drop_last=False,
        dataset=dataset if dataset else args.dataset
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), "Sampler type '{}' not supported".format(
        type(loader.sampler))
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(cur_epoch)


def _dataset_class_num(dataset_name):
    """Query to obtain class nums of datasets."""
    return _NUM_CLASSES_CATALOG[dataset_name]

""" Dataset Factory

Hacked together by / Copyright 2021, Ross Wightman
"""
import os
#import hub
from .fgvc import CUB200Dataset, NabirdsDataset, DogsDataset, FlowersDataset, CarsDataset
from .dtd import DTD
from .gtsrb import GTSRB
from .food import Food101
from .svhn import SVHN
from .vtab import VTAB

from torchvision.datasets import CIFAR100, CIFAR10, MNIST, QMNIST, KMNIST, FashionMNIST, ImageNet, ImageFolder
try:
    from torchvision.datasets import Places365
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    from torchvision.datasets import INaturalist
    has_inaturalist = True
except ImportError:
    has_inaturalist = False

from timm.data.dataset import IterableImageDataset, ImageDataset



# my datasets
from .stanford_dogs import dogs
from .nabirds import NABirds
from .cub2011 import Cub2011
from .vtab import VTAB



_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    qmist=QMNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
)
_TRAIN_SYNONYM = {'train', 'training'}
_EVAL_SYNONYM = {'val', 'valid', 'validation', 'eval', 'evaluation', 'test'}

# _VTAB_DATASET = ['caltech101', 'clevr_count', 'dmlab', 'dsprites_ori', 'eurosat', 'flowers102', 'patch_camelyon', 'smallnorb_azi', 'svhn', 'cifar', 'clevr_dist', 'dsprites_loc', 'dtd', 'kitti', 'pets', 'resisc45', 'smallnorb_ele', 'sun397', 'diabetic_retinopathy']
_VTAB_DATASET = ['caltech101', 'clevr_count', 'diabetic_retinopathy', 'dsprites_loc', 'dtd', 'kitti', 'oxford_iiit_pet', 'resisc45', 'smallnorb_ele', 'svhn', 'cifar', 'clevr_dist', 'dmlab', 'dsprites_ori', 'eurosat', 'oxford_flowers102', 'patch_camelyon', 'smallnorb_azi', 'sun397']



def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root


def create_dataset(
        name,
        root,
        split='val',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=True,
        batch_size=None,
        repeats=0,
        **kwargs
):
    """ Dataset factory method

    In parenthesis after each arg are the type of dataset supported for each arg, one of:
      * folder - default, timm folder (or tar) based ImageDataset
      * torch - torchvision based datasets
      * TFDS - Tensorflow-datasets wrapper in IterabeDataset interface via IterableImageDataset
      * all - any of the above

    Args:
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset (all)
        split: dataset split (all)
        search_split: search for split specific child fold from root so one can specify
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (folder, torch/folder)
        class_map: specify class -> index mapping via text file or dict (folder)
        load_bytes: load data, return images as undecoded bytes (folder)
        download: download dataset if not present and supported (TFDS, torch)
        is_training: create dataset in train mode, this is different from the split.
            For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS)
        batch_size: batch size hint for (TFDS)
        repeats: dataset repeats per iteration i.e. epoch (TFDS)
        **kwargs: other args to pass to dataset

    Returns:
        Dataset object
    """
    if name in _TORCH_BASIC_DS:
        torch_kwargs = dict(root=root, download=download, **kwargs)
        ds_class = _TORCH_BASIC_DS[name]
        use_train = split in _TRAIN_SYNONYM
        dataset = ds_class(train=use_train, **torch_kwargs)
    elif name.startswith('tfds/'):
         dataset = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training,
            download=download, batch_size=batch_size, repeats=repeats, **kwargs)
    elif 'vtab-' in name:
        name = name.split('-', 2)[-1]
        assert name in _VTAB_DATASET, f'The dataset {name} is not in VTAB-1k datasets.'
        root = os.path.join(root, 'vtab-1k', name)
        dataset = VTAB(root=root, train=is_training, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        # define my datasets
        if name == 'stanford_dogs120':
            dataset = dogs(root=root, train=is_training, download=download, **kwargs)

        elif name == 'nabirds1011':
            dataset = NABirds(root=root, train=is_training, **kwargs)

        elif name == 'cub200':
            dataset = Cub2011(root=root, train=is_training, download=download, **kwargs)

        # datasets from SVP
        elif name == "nabirds":
            dataset = NabirdsDataset(root, split=split, transform=None)
             
        elif name == "stanford_dogs":
            dataset = DogsDataset(root, split=split, transform=None)
             
        elif name == "flowers102":
            dataset = FlowersDataset(root, split=split, transform=None)
             
        elif name == "CARS":
            dataset = CarsDataset(root, split=split, transform=None)
             
        elif name == "dtd47":
            dataset = DTD(data_path=root, split=split, transform=None)
             
        elif name == "gtsrb43":
            dataset = GTSRB(data_path=root, split=split, transform=None)
             
        elif name == "food101":
            dataset = Food101(data_path=root, split=split, transform=None)
             
        elif name == "svhn10":
            dataset = SVHN(data_path=root, split=split, transform=None)
             
        else:
            if os.path.isdir(os.path.join(root, split)):
                root = os.path.join(root, split)
            else:
                if search_split and os.path.isdir(root):
                    root = _search_split(root, split)
            dataset = ImageDataset(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
    return  dataset


# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.utils.data as data
from PIL import Image
import os
import os.path
from torchvision import transforms
import torch
from timm.data import create_transform
import numpy as np
from collections import Counter

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
    "gtsrb": 43
}

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self._class_ids = [target for _, target in self.imlist]  # 初始化时计算类标签ID

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)

    def get_class_weights(self, weight_type):
        """Get a list of class weights, return a list of floats."""
        cls_num = len(set(self._class_ids))
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        num_per_cls = np.array([id2counts[i] for i in range(cls_num)])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        else:
            raise ValueError("Unsupported weight_type: {}".format(weight_type))

        weight_list = num_per_cls ** mu
        weight_list = np.divide(weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()


def get_data(data_path, name, evaluate=True, batch_size=32, few_shot=False, shot=1, seed=1, mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]):
    if few_shot:
        root_train = './data/few_shot/' + name + '/images/train/'
        root_val = './data/few_shot/' + name + '/images/test/'
        trainval_flist = './data/few_shot/' + name + '/train_meta.list.num_shot_%d.seed_%d' % (shot, seed)
        train_flist = './data/few_shot/' + name + '/train_meta.list.num_shot_%d.seed_%d' % (shot, seed)
        val_flist = './data/few_shot/' + name + '/test_meta.list'
        test_flist = './data/few_shot/' + name + '/test_meta.list'
        train_transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
    else:
        root = os.path.join(data_path, name)
        root_train = root
        root_val = root
        trainval_flist = root + "/train800val200.txt"
        train_flist = root + "/train800.txt"
        val_flist = root + "/val200.txt"
        test_flist = root + "/test.txt"
        # train_transform = transforms.Compose([
        #     transforms.Resize((224, 224), interpolation=3),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # train_transform = transforms.Compose([
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色增强
        #     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        #     transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),  # 仿射变换
        #     transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # 模糊
        #     transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 随机裁剪
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        if name == "kitti":
            train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.Resize((256, 256), interpolation=3),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif name in ["smallnorb_azi", "smallnorb_ele"]:
            print("using specific transform")
            train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.Resize((256, 256), interpolation=3),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif name in ["dsprites_loc"]:
            print("using specific transform")
            train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.Resize((224, 224), interpolation=3),
                # transforms.Resize((256, 256), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif name in ["dsprites_ori"]:
            print("using specific transform")
            train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                # transforms.Resize((224, 224), interpolation=3),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
                # transforms.Resize((256, 256), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
    if name in ["smallnorb_azi", "smallnorb_ele"]:
        val_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:   
        val_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if evaluate:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_train, flist=trainval_flist,
                          transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_val, flist=test_flist,
                          transform=val_transform),
            batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_train, flist=train_flist,
                          transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_val, flist=val_flist,
                          transform=val_transform),
            batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)

    test_load = None
    return train_loader, val_loader, test_load

def _dataset_class_num(dataset_name):
    """Query to obtain class nums of datasets."""
    return _NUM_CLASSES_CATALOG[dataset_name]

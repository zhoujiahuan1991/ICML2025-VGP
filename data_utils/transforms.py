"""Image transformations."""
import torchvision as tv
from data_utils.autoaugment import AutoAugImageNetPolicy

def get_transforms(split, size, args):
    # if using clip backbones, we adopt clip official normalization.

    # define the sizes used for resizing and cropping
    if size == 448:
        resize_dim = 512
        crop_dim = 448
    elif size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384
        
    # applying different tranforms for training and test
    if split == "train":
        if 'stanford_cars' in args.dataset:
            print('using autoaugment')
            transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                AutoAugImageNetPolicy(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        elif 'vtab' in args.dataset:  
            if args.dataset in ["kitti"]:
                transform = tv.transforms.Compose([
                    tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    tv.transforms.Resize((256, 256), interpolation=3),
                    # tv.transforms.RandomHorizontalFlip(p=0.5),
                    tv.transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif args.dataset in ["smallnorb_azi", "smallnorb_ele"]:
                print("using specific transform")
                transform = tv.transforms.Compose([
                    tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    tv.transforms.Resize((256, 256), interpolation=3),
                    tv.transforms.RandomCrop((224, 224)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif args.dataset in ["dsprites_loc"]:
                print("using specific transform")
                transform = tv.transforms.Compose([
                    tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    tv.transforms.Resize((224, 224), interpolation=3),
                    # tv.transforms.Resize((256, 256), interpolation=3),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif args.dataset in ["dsprites_ori"]:
                print("using specific transform")
                transform = tv.transforms.Compose([
                    tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    # tv.transforms.Resize((224, 224), interpolation=3),
                    tv.transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
                    # tv.transforms.Resize((256, 256), interpolation=3),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = tv.transforms.Compose([
                    tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    tv.transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
                    tv.transforms.RandomHorizontalFlip(p=0.5),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            # HTA training set data augmentation
            transform = tv.transforms.Compose(
                [
                    tv.transforms.Resize(resize_dim),
                    # tv.transforms.RandomCrop(crop_dim),
                    tv.transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
                    tv.transforms.RandomHorizontalFlip(0.5),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    elif split == 'validation':
        if 'vtab-1k' in args.data_dir:
            if args.dataset in ["smallnorb_azi", "smallnorb_ele"]:
                transform = tv.transforms.Compose([
                    tv.transforms.Resize((256, 256), interpolation=3),
                    tv.transforms.CenterCrop((224, 224)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:   
                transform = tv.transforms.Compose([
                    tv.transforms.Resize((224, 224), interpolation=3),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            # HTA test set data augmentation
            transform = tv.transforms.Compose(
                [
                    tv.transforms.Resize(resize_dim),
                    tv.transforms.CenterCrop(crop_dim),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    else:
        raise NotImplementedError("Unrecognized dataset split!")
        
    return transform

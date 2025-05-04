import subprocess
import os
from tqdm import tqdm
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "7" 

# datasets = ['vtab-caltech101', 'vtab-clevr_count', 'vtab-diabetic_retinopathy', 'vtab-dsprites_loc']
# datasets = ['vtab-dtd', 'vtab-kitti', 'vtab-oxford_iiit_pet', 'vtab-resisc45', 'vtab-smallnorb_ele',]
# datasets = ['vtab-svhn', 'vtab-cifar', 'vtab-clevr_dist', 'vtab-dmlab', 'vtab-dsprites_ori',]
# datasets = ['vtab-eurosat', 'vtab-oxford_flowers102', 'vtab-patch_camelyon', 'vtab-smallnorb_azi', 'vtab-sun397']

#2025.2.22
# datasets = ['vtab-kitti', 'vtab-patch_camelyon', 'vtab-smallnorb_ele']
# datasets = ['vtab-cifar', 'vtab-clevr_dist', 'vtab-dmlab', 'vtab-dsprites_ori',]

# datasets = ['vtab-kitti', 'vtab-eurosat', 'vtab-oxford_flowers102', 'vtab-oxford_iiit_pet',]
# datasets = ['vtab-smallnorb_ele'] #'vtab-svhn'
# datasets = ['cub200', 'stanford_dogs120', 'flowers102', 'gtsrb43','svhn10', 'cifar10', 'dtd47', 'nabirds1011', 'food101','cifar100']
datasets = ['cub200', 'stanford_dogs120', 'flowers102', 'gtsrb43','svhn10']
# datasets = ['svhn10', 'cifar10', 'dtd47', 'nabirds1011', 'food101',] #'cifar100'
# datasets = ['cifar100']
def run_command(exp_name="hyper", model="pvig_lor_gp_m_224_gelu", learning_rate=0.00005, batch_size=50, PEFT=True):
    for dataset in tqdm(datasets):
        command=[
                "python",
                "main.py",
                "--exp_name", exp_name,
                "--model", model,
                "--pretrain_path", "pretrained_bases/pvig_m_im21k_90e.pth",
                "--dataset", dataset,
                "--batch_size", str(batch_size),
                "--lr", str(learning_rate)
            ]
        if PEFT:
            command.append("--peft")
        subprocess.run(command, env=env)

run_command(exp_name="greedy-vig-finetune", model="greedy_vig_b", learning_rate=0.00005, batch_size=30, PEFT=False)

# run_command(exp_name="probing-vtab", model="pvig_lor_gp_m_224_gelu", learning_rate=0.0010, batch_size=60, PEFT=True)

# run_command(exp_name="probing-vtab", model="pvig_m_224_gelu", learning_rate=0.0010, batch_size=60, PEFT=False)

# run_command(exp_name="ablation-efficiency", model="pvig_m_224_gelu", learning_rate=0.0010, batch_size=40, PEFT=False)

# run_command(num_layers=2, learning_rate=0.0002)

# run_command(num_layers=1, learning_rate=0.0002)

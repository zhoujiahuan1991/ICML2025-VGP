import subprocess
import os
from tqdm import tqdm
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "5" 

# datasets = ["toxcast", "sider", "clintox", "muv", "hiv", "bace", "bbbp", "tox21"]
datasets = ["sider", "muv",]

def run_command(num_layers=3, learning_rate=0.0002):
    for dataset in tqdm(datasets):
        command=[
            "python",
            "vision_graph_prompt_tuning_full_shot.py",
            "--model_file", "pretrained_models/edgepred.pth",
            "--dataset", dataset,
            "--tuning_type", "vgp",
            "--num_layers", str(num_layers),
            "--lr", str(learning_rate)
        ]
        subprocess.run(command, env=env)


# run_command(num_layers=3, learning_rate=0.0002)

# run_command(num_layers=2, learning_rate=0.0002)

# run_command(num_layers=1, learning_rate=0.0002)

run_command(num_layers=3, learning_rate=0.0001)

run_command(num_layers=2, learning_rate=0.0001)

run_command(num_layers=1, learning_rate=0.0001)

"""
train_sweeps.py runs hyperparameter search using Weights and Biases and the predefined sweep.yml file
Please adapt the sweep.yml file to your needs and run the script with the following command:
    python scripts/train_sweeps.py --sweep_file sweep.yml

For more information on sweeps in Weights and Biases, please refer to the following link:
https://docs.wandb.ai/guides/sweeps
"""
import wandb
from pathlib import Path
import yaml
import argparse
from .train import train, ESDConfig
import torch

torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_device('cuda')

def main():
    wandb.init(project="CNN-sweep")
    print(wandb.config)
    options = ESDConfig(**wandb.config)
    train(options)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps using Weights and Biases")

    parser.add_argument('--sweep_file', type=str, help="Path to sweep.yml file", default=None)
    
    parse_args = parser.parse_args()
    
    if parse_args.sweep_file is not None:
        with open(Path(parse_args.sweep_file)) as f:
            sweep_config = yaml.safe_load(f)
            print(f"Sweep config: {sweep_config}")

        sweep_id = wandb.sweep(sweep=sweep_config, project="CNN-sweep")
        wandb.agent(sweep_id, function=main, count=1)
"""
train_sweeps.py runs hyperparameter search using Weights and Biases and the predefined sweep.yml file
    python scripts/train_sweeps.py --sweep_file sweep.yml
    multiple sweeps can be run in parallel by running the above command in multiple terminals

For more information on sweeps in Weights and Biases, please refer to the following link:
https://docs.wandb.ai/guides/sweeps
"""
import wandb
from pathlib import Path
import yaml
import argparse
from .train import train, ESDConfig
import torch

# check for GPU, and use if available
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_device('cuda')

def main():
    # Initialize weights and biases logger with the correct project name
    wandb.init(project="sweep")
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

        # run sweep via main() function, make sure the project name is correct
        sweep_id = wandb.sweep(sweep=sweep_config, project="sweep")
        wandb.agent(sweep_id, function=main, count=100)
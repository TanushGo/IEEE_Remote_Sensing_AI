### python -m scripts.train_unet_sweep

import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))

import torch
import pytorch_lightning as pl

import os
from dataclasses import dataclass

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation

import wandb
from lightning.pytorch.loggers import WandbLogger

# check for GPU, and use if available
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_device('cuda')

@dataclass
class ESDConfig:
    """
    IMPORTANT: This class is used to define the configuration for the experiment
    Please make sure to use the correct types and default values for the parameters
    and that the path for processed_dir contain the tiles you would like 
    """
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    selected_bands: None = None
    model_type: str = "UNet"
    tile_size_gt: int = 4
    batch_size: int = 8
    max_epochs: int = 2
    seed: int = 12378921
    learning_rate: float = 0.00030
    num_workers: int = 11
    accelerator: str = "gpu"
    devices: int = 1
    in_channels: int = 99
    out_channels: int = 4
    depth: int = 2
    n_encoders: int = 2
    embedding_size: int = 64
    pool_sizes: str = '5,5,2' # List[int] = [5,5,2]
    kernel_size: int = 3
    scale_factor: int = 50
    wandb_run_name: str | None = None

def train(options: ESDConfig):
    """
    Prepares datamodule and model, then runs the training loop

    Inputs:
        options: ESDConfig
            options for the experiment
    """

    # Initialize weights and biases logger
    wandb.init(project="UNET", name=options.wandb_run_name, config=options.__dict__)
    wandb_logger = WandbLogger(project="UNET")
    
    # initiate the ESDDatamodule with the options object
    # prepare_data in case the data has not been preprocessed
    esd_dm = ESDDataModule(options.processed_dir, options.raw_dir, options.selected_bands, options.tile_size_gt, options.batch_size, options.seed)
    esd_dm.prepare_data()

    # create a dictionary with the parameters to pass to the models
    params = options.__dict__

    # initialize the ESDSegmentation module
    esd_segmentation = ESDSegmentation(options.model_type, options.in_channels, options.out_channels, options.learning_rate, params)

    # Callbacks for the training loop
    # ModelCheckpoint: saves intermediate results for the neural network in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=root / 'models' / options.model_type,
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            every_n_train_steps=1000
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    # create a pytorch_lightning Trainer

    # If using a GPU, use the first two lines, otherwise use the third line
    # torch.set_float32_matmul_precision('medium')
    # trainer = pl.Trainer(callbacks=callbacks, max_epochs=options.max_epochs, devices=options.devices, accelerator=options.accelerator, logger=wandb_logger)
    trainer = pl.Trainer(callbacks=callbacks, max_epochs=options.max_epochs, logger=wandb_logger)

    # run trainer.fit with the datamodule option
    trainer.fit(esd_segmentation, datamodule=esd_dm)

def main():
    # GPU check and statistic for testing
    if torch.cuda.is_available():
        print(f'device count: {torch.cuda.device_count()}')
        print(f'current device: {torch.cuda.current_device()}')
        print(f'device name: {torch.cuda.get_device_name()}')
        torch.set_default_device('cuda')

    # Initialize weights and biases logger with the correct project name
    wandb.init(project="UNET-sweep")
    print(wandb.config)
    options = ESDConfig(**wandb.config)
    train(options)


if __name__ == '__main__':

    # sweep parameters (.yml file had issues originally)
    sweep_config = {
        'name': 'UNet-sweep',
        'method': 'bayes',
        'metric': {
            'name': 'core_values/average ACC',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.00001,
                'max': 0.1,
                'distribution': 'log_uniform_values'
            },
            'batch_size': {
                'values': [4, 8, 16]
            },
            'max_epochs': {
                'min': 3,
                'max': 8,
                'distribution': 'int_uniform'
            },
            'n_encoders': {
                'values': [2, 3, 4]
            },
            'scale_factor': {
                'values': [25, 50, 100]
            }
        }
    }

    # run sweep via main() function, make sure the project name is correct
    sweep_id = wandb.sweep(sweep=sweep_config, project="UNet-sweep")
    wandb.agent(sweep_id, function=main, count=100)
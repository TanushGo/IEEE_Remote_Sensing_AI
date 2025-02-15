import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
import numpy as np

from src.models.supervised.random_forests import RandomForests
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.metrics import accuracy_score

# check for GPU, and use if available
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.set_default_device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model_type: str = "SegmentationCNN"
    tile_size_gt: int = 4
    batch_size: int = 31
    max_epochs: int = 1
    seed: int = 12378921
    learning_rate: float = 0.0001584
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

    # Initialize the weights and biases logger
    wandb.init(project="RandomForests", name=options.wandb_run_name, config=options.__dict__)
    wandb_logger = WandbLogger(project="RandomForests")

    
    # initiate the ESDDatamodule with the options object
    # prepare_data in case the data has not been preprocessed
    esd_dm = ESDDataModule(options.processed_dir, options.raw_dir, options.selected_bands, options.tile_size_gt, options.batch_size, options.seed)
    esd_dm.prepare_data()
    
    esd_dm.setup(stage="fit")

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

    # trainer.fit(esd_segmentation, datamodule=esd_dm)


    #preload model if it exists
    model = ESDSegmentation.load_from_checkpoint(root / "models" / "RandomForests" / "last.ckpt")


   

#do feature extraction on the output from putting data into the model this will create the data for random forest training
    def extract_features(model, data_loader):
        CNN_features = torch.nn.Sequential(*list(model.children())[:-1])
        #resnet_features = model.backbone
        model.eval()
        CNN_features.eval()
        features_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                #features = model(inputs.float().to(device)).cpu()#.numpy()
                # print(type(features))
                features = CNN_features(inputs.float().to(device)).cpu()
                print(f"Features are {features.shape}")
                flattened_features = torch.permute(features,(1,0,2,3)).reshape(-1, features.shape[1])#.transpose(1, 0)
                print(flattened_features.shape)
                features_list.append(flattened_features.numpy())
                labels_flatten = labels.flatten()
                print("labels flatten shape",labels_flatten.shape)
                print("labels",labels.shape)
                labels_list.append(labels_flatten.cpu().numpy())

        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)
        print(features.shape)
        print(labels.shape)
        train_class = list(np.unique(labels))
        print(train_class)
        return features, labels

   
    #getting train and test features from the dataset
    train_features, train_labels = extract_features(model, esd_dm.train_dataloader())
    test_features, test_labels = extract_features(model, esd_dm.val_dataloader())

    # Train Random Forest classifier using extracted features
    random_forest_classifier = RandomForestClassifier(n_estimators=500, criterion="log_loss", random_state=42, max_features=4,
                                                      min_samples_leaf=100)
    random_forest_classifier.fit(train_features, train_labels)

    predictions = random_forest_classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print("Random Forest Classifier Accuracy:", accuracy)
    

    


if __name__ == '__main__':

    # load dataclass arguments from yml file
    config = ESDConfig()
    parser = ArgumentParser()
    
    parser.add_argument("--model_type", type=str, help="The model to initialize.", default=config.model_type)
    parser.add_argument("--learning_rate", type=float, help="The learning rate for training model", default=config.learning_rate)
    parser.add_argument("--max_epochs", type=int, help="Number of epochs to train for.", default=config.max_epochs)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument('--batch_size', help="Batch size to train in", type=int, default=config.batch_size)

    parser.add_argument('--in_channels', type=int, default=config.in_channels, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=config.out_channels, help='Number of output channels')
    parser.add_argument('--depth', type=int, help="Depth of the encoders (CNN only)", default=config.depth)
    parser.add_argument('--n_encoders', type=int, help="Number of encoders (Unet only)", default=config.n_encoders)
    parser.add_argument('--embedding_size', type=int, help="Embedding size of the neural network (CNN/Unet)", default=config.embedding_size)
    parser.add_argument('--pool_sizes', help="A comma separated list of pool_sizes (CNN only)", type=str, default=config.pool_sizes)
    parser.add_argument('--kernel_size', help="Kernel size of the convolutions", type=int, default=config.kernel_size)
    parser.add_argument('--scale_factor', help="Scale factor between the labels and the image (Unet and Transfer Resnet)", type=int, default=config.scale_factor)

    # --pool_sizes=5,5,2 to call it correctly
    
    parse_args = parser.parse_args()
    
    train(ESDConfig(**parse_args.__dict__))
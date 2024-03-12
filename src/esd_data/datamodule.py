""" This module contains the PyTorch Lightning ESDDataModule to use with the
PyTorch ESD dataset."""

import pytorch_lightning as pl
from torch import Generator
from torch.utils.data import DataLoader, random_split
import torch
from .dataset import DSE
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ..preprocessing.subtile_esd_hw02 import grid_slice
from ..preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_viirs,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
)
from ..preprocessing.file_utils import load_satellite
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)
from torchvision import transforms
from copy import deepcopy
from typing import List, Tuple, Dict
from src.preprocessing.file_utils import Metadata
from sklearn.model_selection import train_test_split

def collate_fn(batch):
    Xs = []
    ys = []
    metadatas = []
    for X, y, metadata in batch:
        if len(X.shape) == 4:
            X = X.squeeze(0)
            # X = X.view(X.shape[0], -1)
        Xs.append(X)
        ys.append(y)
        metadatas.append(metadata)

    # Xs = np.stack(Xs)
    # ys = np.stack(ys)
    return torch.stack(Xs), torch.stack(ys), metadatas


class ESDDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning ESDDataModule to use with the PyTorch ESD dataset.

    Attributes:
        processed_dir: str | os.PathLike
            Location of the processed data
        raw_dir: str | os.PathLike
            Location of the raw data
        selected_bands: Dict[str, List[str]] | None
            Dictionary mapping satellite type to list of bands to select
        tile_size_gt: int
            Size of the ground truth tiles
        batch_size: int
            Batch size
        seed: int
            Seed for the random number generator
    """

    def __init__(
        self,
        processed_dir: str | os.PathLike,
        raw_dir: str | os.PathLike,
        selected_bands: Dict[str, List[str]] | None = None,
        tile_size_gt=4,
        batch_size=32,
        seed=12378921,
    ):

        # set transform to a composition of the following transforms:
        # AddNoise, Blur, RandomHFlip, RandomVFlip, ToTensor
        # utilize the RandomApply transform to apply each of the transforms
        # with a probability of 0.5
        self.train_tile_paths = []
        self.val_tile_paths = []
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.selected_bands = selected_bands
        self.tile_size_gt = tile_size_gt
        self.batch_size = batch_size
        self.seed = seed

        # added these 3 bc idk how else to make it work
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

        torch.manual_seed(seed)

        self.transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        AddNoise(),
                        Blur(),
                        RandomHFlip(),
                        RandomVFlip(),
                    ],
                    p=0.5,
                ),
                ToTensor(),
            ]
        )

    def __load_and_preprocess(
        self,
        tile_dir: str | os.PathLike,
        satellite_types: List[str] = [
            "viirs",
            "sentinel1",
            "sentinel2",
            "landsat",
            "gt",
        ],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Metadata]]]:
        """
        Performs the preprocessing step: for a given tile located in tile_dir,
        loads the tif files and preprocesses them just like in homework 1.

        Input:
            tile_dir: str | os.PathLike
                Location of raw tile data
            satellite_types: List[str]
                List of satellite types to process

        Output:
            satellite_stack: Dict[str, np.ndarray]
                Dictionary mapping satellite_type -> (time, band, width, height) array
            satellite_metadata: Dict[str, List[Metadata]]
                Metadata accompanying the statellite_stack
        """
        preprocess_functions = {
            "viirs": preprocess_viirs,
            "sentinel1": preprocess_sentinel1,
            "sentinel2": preprocess_sentinel2,
            "landsat": preprocess_landsat,
            "gt": lambda x: x,
        }

        satellite_stack = {}
        satellite_metadata = {}
        for satellite_type in satellite_types:
            stack, metadata = load_satellite(tile_dir, satellite_type)

            stack = preprocess_functions[satellite_type](stack)

            satellite_stack[satellite_type] = stack
            satellite_metadata[satellite_type] = metadata

        satellite_stack["viirs_maxproj"] = np.expand_dims(
            maxprojection_viirs(satellite_stack["viirs"], clip_quantile=0.0), axis=0
        )
        satellite_metadata["viirs_maxproj"] = deepcopy(satellite_metadata["viirs"])
        for metadata in satellite_metadata["viirs_maxproj"]:
            metadata.satellite_type = "viirs_maxproj"

        return satellite_stack, satellite_metadata

    # def prepare_data(self):
    #     """
    #     If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory)

    #     For each tile,
    #         - load and preprocess the data in the tile
    #         - grid slice the data
    #         - for each resulting subtile
    #             - save the subtile data to self.processed_dir
    #     """
    #     # if the processed_dir does not exist, process the data and create
    #     # subtiles of the parent image to save
    #     if not os.path.exists(self.processed_dir):

    #         # fetch all the parent images in the raw_dir
    #         # for each parent image in the raw_dir
    #         for img in Path(self.raw_dir).glob("*"):

    #             # call __load_and_preprocess to load and preprocess the data for all satellite types
    #             satellite_stack, satellite_metadata = self.__load_and_preprocess(img)

    #             # grid slice the data with the given tile_size_gt
    #             subtile_stack = grid_slice(
    #                 satellite_stack, satellite_metadata, self.tile_size_gt
    #             )

    #             # save each subtile
    #             for subtile in subtile_stack:
    #                 subtile.save(self.processed_dir)

    # def setup(self, stage: str):
    #     """
    #     Create self.train_dataset and self.val_dataset.0000ff

    #     Hint: Use torch.utils.data.random_split to split the Train
    #     directory loaded into the PyTorch dataset DSE into an 80% training
    #     and 20% validation set. Set the seed to 1024.
    #     """

    #     # if the stage is "fit", load the data from the processed_dir
    #     if stage == "fit":
    #         # load the data from the processed_dir
    #         # and split the data into 80% training and 20% validation
    #         # set the seed to 1024

    #         dataset = DSE(self.processed_dir, transform=self.transform)
    #         train_size = int(0.8 * len(dataset))
    #         val_size = len(dataset) - train_size
    #         self.train_dataset, self.val_dataset = random_split(
    #             dataset, [train_size, val_size], generator=Generator().manual_seed(1024)
    #         )

    def train_dataloader(self):
        """
        Create and return a torch.utils.data.DataLoader with
        self.train_dataset
        """

        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )

    def val_dataloader(self):
        """
        Create and return a torch.utils.data.DataLoader with
        self.val_dataset
        """

        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )





# THINGS TO CHANGE: 

# STEP 1: Add attributes to hold train and validation tile paths

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.train_tile_paths = []
    #     self.val_tile_paths = []
        
# STEP 2: Split parent tiles into train and validation sets before processing

    def prepare_data(self):
        if not os.path.exists(self.processed_dir):
            parent_tiles = list(Path(self.raw_dir).glob("*"))
            train_tiles, val_tiles = train_test_split(parent_tiles, test_size=0.2, random_state=self.seed)

            # Process and save train tiles
            for tile in train_tiles:
                self.__process_and_save_tile(tile, "Train")

            # Process and save validation tiles
            for tile in val_tiles:
                self.__process_and_save_tile(tile, "Val")
            
# STEP 3: Adapting processing logic that saves subtiles to Train or Val directories

    def __process_and_save_tile(self, tile_path, dataset_type):
        
        satellite_stack, satellite_metadata = self.__load_and_preprocess(tile_path)
        subtile_stack = grid_slice(satellite_stack, satellite_metadata, self.tile_size_gt)

        # Save each subtile in the corresponding directory
        save_dir = Path(self.processed_dir) / str(self.tile_size_gt) / dataset_type
        save_dir.mkdir(parents=True, exist_ok=True)
        for subtile in subtile_stack:
            # print(f'Saving {subtile.satellite_stack} to {save_dir}')
            subtile.save(save_dir)

# STEP 4:  Setup logic that loads from Train/Val directories instead of a single directory

    def setup(self, stage: str):
        if stage == "fit":
            train_dataset_path = Path(self.processed_dir) / str(self.tile_size_gt) / "Train"
            val_dataset_path = Path(self.processed_dir) / str(self.tile_size_gt) / "Val"
            self.train_dataset = DSE(train_dataset_path, transform=self.transform)
            self.val_dataset = DSE(val_dataset_path, transform=self.transform)
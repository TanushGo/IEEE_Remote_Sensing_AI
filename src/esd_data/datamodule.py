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
    '''
    extracts and returns the Xs, ys, and metadata from batch data
    :param batch: list of tuples in the form (X, y, metadata)
    :return: torch stack of Xs, torch stack of ys, list of metaata
    '''
    Xs = []
    ys = []
    metadatas = []
    for X, y, metadata in batch:
        if len(X.shape) == 4:
            X = X.squeeze(0)
        Xs.append(X)
        ys.append(y)
        metadatas.append(metadata)

    # explicitly converting to torch float 32 due to mac compatibility issues
    return torch.stack(Xs).to(dtype=torch.float32), torch.stack(ys).to(dtype=torch.float32), metadatas


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
        self._log_hyperparams = True
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

    # Split parent tiles into train and validation sets before processing
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

    # Adapting processing logic that saves subtiles to Train or Val directories
    def __process_and_save_tile(self, tile_path, dataset_type):
        
        satellite_stack, satellite_metadata = self.__load_and_preprocess(tile_path)
        subtile_stack = grid_slice(satellite_stack, satellite_metadata, self.tile_size_gt)

        # Save each subtile in the corresponding directory
        save_dir = Path(self.processed_dir) / str(self.tile_size_gt) / dataset_type
        save_dir.mkdir(parents=True, exist_ok=True)
        for subtile in subtile_stack:
            subtile.save(save_dir)

    # Setup logic that loads from Train/Val directories instead of a single directory
    def setup(self, stage: str):
        if stage == "fit":
            train_dataset_path = Path(self.processed_dir) / str(self.tile_size_gt) / "Train"
            val_dataset_path = Path(self.processed_dir) / str(self.tile_size_gt) / "Val"
            self.train_dataset = DSE(train_dataset_path, transform=self.transform)
            self.val_dataset = DSE(val_dataset_path, transform=self.transform)

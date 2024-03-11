import pyprojroot
import sys
import os

root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import restitch_eval, restitch_and_plot
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tifffile


@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / "data/processed/4x4"
    raw_dir: str | os.PathLike = root / "data/raw/Train"
    results_dir: str | os.PathLike = root / "data/predictions" / "FCNResnetTransfer"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "FCNResnetTransfer" / "last.ckpt"


def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Load datamodule
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
        tile_size_gt=options.tile_size_gt,
        batch_size=options.batch_size,
        seed=options.seed,
    )

    # load model from checkpoint at options.model_path
    datamodule.setup("test")
    model = ESDSegmentation.load_from_checkpoint(checkpoint_path=options.model_path)

    # set the model to evaluation mode (model.eval())
    model.eval()

    # this is important because if you don't do this, some layers
    # will not evaluate properly

    # instantiate pytorch lightning trainer
    # Instantiate PyTorch Lightning trainer
    trainer = pl.Trainer(
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(dirpath=options.results_dir, save_top_k=1),
            RichProgressBar(),
            RichModelSummary(),
        ],
        gpus=1 if torch.cuda.is_available() else 0,
    )

    # run the validation loop with trainer.validate
    # Run the validation loop with trainer.validate
    trainer.validate(model, datamodule=datamodule)

    # run restitch_and_plot

    # for every subtile in options.processed_dir/Val/subtiles
    val_dir = Path(options.processed_dir) / "Val" / "subtiles"
    results_dir = Path(options.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    # run restitch_eval on that tile followed by picking the best scoring class
    # save the file as a tiff using tifffile
    # save the file as a png using matplotlib
    # tiles = ...
    for parent_tile_id in val_dir.glob("*"):
        satellite_type = "sentinel2"  # Example, adjust based on your use case
        rgb_bands = [3, 2, 1]  # Adjust based on your satellite type

        # Running restitch_and_plot for visualization
        restitch_and_plot(
            options,
            datamodule,
            model,
            parent_tile_id.stem,
            satellite_type,
            rgb_bands,
            image_dir=results_dir,
        )

        # Assuming the structure for the evaluation includes retrieving range_x and range_y dynamically
        range_x, range_y = (0, 5), (
            0,
            5,
        )  # Example ranges, adjust based on your actual data

        # Running restitch_eval for evaluation
        stitched_image, stitched_gt, stitched_pred = restitch_eval(
            dir=val_dir,
            satellite_type=satellite_type,
            tile_id=parent_tile_id.stem,
            range_x=range_x,
            range_y=range_y,
            datamodule=datamodule,
            model=model,
        )

        # Saving the predicted image as TIFF
        tifffile.imwrite(
            results_dir / f"{parent_tile_id.stem}_prediction.tif", stitched_pred
        )

        # Plotting and saving the predicted image as PNG
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Settlements", np.array(["#ff0000", "#0000ff", "#ffff00", "#b266ff"]), N=4
        )
        plt.imshow(stitched_pred, vmin=-0.5, vmax=3.5, cmap=cmap)
        plt.axis("off")
        plt.savefig(
            results_dir / f"{parent_tile_id.stem}_prediction.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    """
    for parent_tile_id in tiles:

        # freebie: plots the predicted image as a jpeg with the correct colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Settlements", np.array(["#ff0000", "#0000ff", "#ffff00", "#b266ff"]), N=4
        )
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(y_pred, vmin=-0.5, vmax=3.5, cmap=cmap)
        plt.savefig(options.results_dir / f"{parent_tile_id}.png")
    """


if __name__ == "__main__":
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(EvalConfig(**parser.parse_args().__dict__))

import pyprojroot
import sys
import os
import re

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
    results_dir: str | os.PathLike = root / "data" / "predictions"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "SegmentationCNN" / "last-v1.ckpt"


def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
        tile_size_gt=options.tile_size_gt,
        batch_size=options.batch_size,
        seed=options.seed,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    model = ESDSegmentation.load_from_checkpoint(checkpoint_path=options.model_path)
    model.eval()

    trainer = pl.Trainer(
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(monitor="val_loss"),
            RichProgressBar(),
            RichModelSummary(),
        ],
        logger=False,  # Disable logging to avoid unnecessary output
    )

    # trainer.validate(model, datamodule=datamodule)

    # Assuming a method to get validation tile IDs from the datamodule
    val_tile_ids = set()
    for path in Path(options.processed_dir / "4" / "Val" / "subtiles").rglob("*.npz"):
        id = re.split("(Tile\d+)", str(path))[1]
        val_tile_ids.add(id)

    
    val_tile_ids = sorted(list(val_tile_ids))
    print(val_tile_ids)

            # Use restitch_and_plot for visualization
    restitch_and_plot(
        options,
        datamodule,
        model,
        "Tile40",
        satellite_type="sentinel2",
        rgb_bands=[3, 2, 1],
        image_dir=options.results_dir,
    )

    for parent_tile_id in val_tile_ids:
        # Use restitch_and_plot for visualization
        # restitch_and_plot(
        #     options,
        #     datamodule,
        #     model,
        #     parent_tile_id,
        #     satellite_type="sentinel2",
        #     rgb_bands=[3, 2, 1],
        #     image_dir=options.results_dir,
        # )

        stitched_image, stitched_ground_truth, stitched_predictions = (
            restitch_eval(
                dir=options.processed_dir,
                satellite_type="sentinel2",
                tile_id=parent_tile_id,
                range_x=(0, 4),  # Example ranges, adjust as necessary
                range_y=(0, 4),
                datamodule=datamodule,
                model=model,
            )
        )
        print("pred",stitched_predictions)

        print("gti",stitched_ground_truth)

        # Save predictions as TIFF
        tifffile.imwrite(
            Path(options.results_dir) / f"{parent_tile_id}_prediction.tif",
            stitched_predictions,
        )

        # Plot and save predictions as PNG
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Settlements", ["#ff0000", "#0000ff", "#ffff00", "#b266ff"], N=4
        )
        fig, ax = plt.subplots()
        ax.imshow(stitched_predictions, vmin=-0.5, vmax=3.5, cmap=cmap)
        plt.savefig(Path(options.results_dir) / f"{parent_tile_id}_prediction.png")
        plt.close(fig)

        figs, axes = plt.subplots()
        axes.imshow(stitched_ground_truth, vmin=-0.5, vmax=3.5, cmap=cmap)
        plt.savefig(Path(options.results_dir) / f"{parent_tile_id}_gt.png")
        plt.close(figs)


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
        "-p",
        "--processed_dir",
        type=str,
        default=config.processed_dir,
        help="Path to processed directory",
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results directory"
    )
    main(EvalConfig(**parser.parse_args().__dict__))

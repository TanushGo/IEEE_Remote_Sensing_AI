import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile


def restitch_and_plot(
    options,
    datamodule,
    model,
    parent_tile_id,
    satellite_type="sentinel2",
    rgb_bands=[3, 2, 1],
    image_dir: None | str | os.PathLike = None,
):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    Args:
        options: EvalConfig
        datamodule: ESDDataModule
        model: ESDSegmentation
        parent_tile_id: str
        satellite_type: str
        rgb_bands: List[int]
    """
    # RGB Satellite Image
    rgb_image = np.stack([stitched_satellite[band - 1] for band in rgb_bands], axis=-1)
    rgb_image = np.clip(rgb_image / np.max(rgb_image), 0, 1)  # Normalize for plotting

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "Settlements", np.array(["#ff0000", "#0000ff", "#ffff00", "#b266ff"]), N=4
    )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(rgb_image)
    axs[0].set_title("RGB Satellite Image")
    axs[0].axis("off")

    # make sure to use cmap=cmap, vmin=-0.5 and vmax=3.5 when running
    # axs[i].imshow on the 1d images in order to have the correct
    # colormap for the images.
    # On one of the 1d images' axs[i].imshow, make sure to save its output as
    # `im`, i.e, im = axs[i].imshow
    im = axs[1].imshow(stitched_ground_truth, cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(stitched_prediction, cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[2].set_title("Model Prediction")
    axs[2].axis("off")

    # The following lines sets up the colorbar to the right of the images
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(
        [
            "Sttlmnts Wo Elec",
            "No Sttlmnts Wo Elec",
            "Sttlmnts W Elec",
            "No Sttlmnts W Elec",
        ]
    )
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "restitched_visible_gt_predction.png")
        plt.close()


import numpy as np
from pathlib import Path
import torch  # Assuming PyTorch is used for model evaluation


def restitch_eval(
    dir: str | os.PathLike,
    satellite_type: str,
    tile_id: str,
    range_x: Tuple[int, int],
    range_y: Tuple[int, int],
    datamodule,
    model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List]]:
    """
    Given a directory of processed subtiles, a tile_id, and a satellite_type,
    this function retrieves tiles between specified ranges to stitch them back
    into their original image. It evaluates these tiles with the given model,
    stitches the ground truth and predictions together, and aggregates metadata.

    Args:
        dir: Directory where the subtiles are saved.
        satellite_type: Type of satellite imagery to be stitched.
        tile_id: Identifier for the tile being processed.
        range_x: Range of tiles on the width dimension.
        range_y: Range of tiles on the height dimension.
        datamodule: Data module containing the dataset and preprocessing logic.
        model: Model to evaluate the tiles.

    Returns:
        stitched_satellite: The stitched satellite imagery.
        stitched_ground_truth: The stitched ground truth labels.
        stitched_prediction: The stitched model predictions.
        satellite_metadata_from_subtile: Aggregated metadata from all subtiles.
    """

    dir = Path(dir)
    satellite_subtile = []
    ground_truth_subtile = []
    predictions_subtile = []
    satellite_metadata_from_subtile = []

    for i in range(*range_x):
        satellite_subtile_row = []
        ground_truth_subtile_row = []
        predictions_subtile_row = []
        satellite_metadata_from_subtile_row = []

        for j in range(*range_y):
            subtile_path = dir / "subtiles" / f"{tile_id}_{i}_{j}.npz"
            subtile = Subtile.load(
                subtile_path
            )  # Load subtile, ensure Subtile has a .load() method

            satellite_data = subtile.satellite_stack[satellite_type]
            ground_truth = subtile.ground_truth  # Assumes ground_truth attribute exists

            # Prepare data for model evaluation
            X = torch.tensor(satellite_data).unsqueeze(0).float()
            if torch.cuda.is_available():
                X = X.cuda()

            # Evaluate with model
            model.eval()
            with torch.no_grad():
                predictions = model(X)
            predictions = predictions.squeeze().cpu().numpy()

            satellite_subtile_row.append(satellite_data)
            ground_truth_subtile_row.append(ground_truth)
            predictions_subtile_row.append(predictions)
            satellite_metadata_from_subtile_row.append(
                subtile.metadata
            )  # Assuming metadata attribute

        satellite_subtile.append(np.concatenate(satellite_subtile_row, axis=-1))
        ground_truth_subtile.append(np.concatenate(ground_truth_subtile_row, axis=-1))
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
        satellite_metadata_from_subtile.append(satellite_metadata_from_subtile_row)

    stitched_satellite = np.concatenate(satellite_subtile, axis=-2)
    stitched_ground_truth = np.concatenate(ground_truth_subtile, axis=-2)
    stitched_predictions = np.concatenate(predictions_subtile, axis=-2)

    return (
        stitched_satellite,
        stitched_ground_truth,
        stitched_predictions,
        satellite_metadata_from_subtile,
    )

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
    # Assuming datamodule has a method to load and preprocess data for a specific tile
    satellite_stack, _ = datamodule.load_and_preprocess(
        parent_tile_id, satellite_types=[satellite_type]
    )
    rgb_image = np.transpose(
        satellite_stack[satellite_type][:, rgb_bands, :, :], (1, 2, 0)
    )
    ground_truth, _ = datamodule.load_and_preprocess(
        parent_tile_id, satellite_types=["gt"]
    )
    # Normalize for display
    rgb_image = rgb_image / rgb_image.max()

    # Model prediction
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        satellite_tensor = torch.from_numpy(satellite_stack[satellite_type]).unsqueeze(
            0
        )  # Add batch dimension
        prediction = model(satellite_tensor.float()).squeeze(
            0
        )  # Remove batch dimension
    prediction = torch.argmax(prediction, dim=0).numpy()  # Convert to class indices

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "Settlements", ["#ff0000", "#0000ff", "#ffff00", "#b266ff"], N=4
    )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axs[0].imshow(rgb_image)
    axs[0].set_title("RGB Satellite Image")
    im = axs[1].imshow(ground_truth.squeeze(), cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[1].set_title("Ground Truth")
    axs[2].imshow(prediction, cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[2].set_title("Model Prediction")

    # Colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(
        [
            "Sttlmnts Wo Elec",
            "No Sttlmnts Wo Elec",
            "Sttlmnts W Elec",
            "No Sttlmnts W Elec",
        ]
    )

    if image_dir is not None:
        plt.savefig(Path(image_dir) / f"{parent_tile_id}_comparison.png")
    else:
        plt.show()
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

    stitched_image = None
    stitched_ground_truth = None
    stitched_predictions = None

    model.eval()  # Ensure the model is in evaluation mode
    for x in range(range_x[0], range_x[1]):
        row_images = []
        row_ground_truths = []
        row_predictions = []
        for y in range(range_y[0], range_y[1]):
            # Load and preprocess data for the specific subtile
            satellite_stack, satellite_metadata = datamodule.load_and_preprocess(
                f"{tile_id}_{x}_{y}", satellite_types=[satellite_type, "gt"]
            )

            # Prepare data for model prediction
            satellite_tensor = torch.from_numpy(
                satellite_stack[satellite_type]
            ).unsqueeze(
                0
            )  # Add batch dimension
            with torch.no_grad():
                prediction = model(satellite_tensor.float()).squeeze(
                    0
                )  # Remove batch dimension
            prediction = torch.argmax(
                prediction, dim=0
            ).numpy()  # Convert to class indices

            # Concatenate data for the current row
            row_images.append(satellite_stack[satellite_type])
            row_ground_truths.append(satellite_stack["gt"])
            row_predictions.append(prediction)

        # Concatenate the rows
        if stitched_image is None:
            stitched_image = np.concatenate(row_images, axis=2)
            stitched_ground_truth = np.concatenate(row_ground_truths, axis=2)
            stitched_predictions = np.concatenate(row_predictions, axis=2)
        else:
            stitched_image = np.concatenate(
                [stitched_image, np.concatenate(row_images, axis=2)], axis=1
            )
            stitched_ground_truth = np.concatenate(
                [stitched_ground_truth, np.concatenate(row_ground_truths, axis=2)],
                axis=1,
            )
            stitched_predictions = np.concatenate(
                [stitched_predictions, np.concatenate(row_predictions, axis=2)], axis=1
            )

    return stitched_image, stitched_ground_truth, stitched_predictions

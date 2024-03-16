import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile
 

def load_npz_data(npz_path, satellite_type, rgb_bands):
    """Load and return RGB image and ground truth from an NPZ file."""
    data = np.load(npz_path)
    rgb_image = np.stack(
        [data[f"{satellite_type}_band_{b}"] for b in rgb_bands], axis=-1
    )
    ground_truth = data["gt"]  # Assuming ground truth is stored with the key 'gt'
    return rgb_image, ground_truth


def restitch_and_plot(
    options, datamodule, model, npz_paths, rgb_bands=[3, 2, 1], image_dir=None
):
    """Plot RGB image, ground truth, and model prediction."""
    for npz_path in npz_paths:
        npz_path_full = Path(datamodule.processed_dir) / "Val" / "subtiles" / npz_path
        rgb_image, ground_truth = load_npz_data(
            npz_path_full, options.satellite_type, rgb_bands
        )

        # Preprocess data as required for the model
        satellite_tensor = (
            torch.tensor(rgb_image.transpose(2, 0, 1)).unsqueeze(0).float()
        )  # Add batch dim and convert to float

        # Prediction
        model.eval()
        with torch.no_grad():
            prediction = (
                model(satellite_tensor).squeeze().argmax(0).numpy()
            )  # Assuming model outputs class probabilities for each pixel

        # Visualization
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(rgb_image / rgb_image.max())  # Normalize for plotting
        axs[0].set_title("RGB Satellite Image")
        axs[1].imshow(ground_truth, cmap="viridis")
        axs[1].set_title("Ground Truth")
        axs[2].imshow(prediction, cmap="viridis")
        axs[2].set_title("Model Prediction")

        if image_dir:
            plt.savefig(Path(image_dir) / f"{Path(npz_path).stem}_comparison.png")
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

""" This code preprocesses the ESD data by stacking the spectral bands
and tiles the images into subimages that are NxN. ""
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from ..preprocessing.file_utils import Metadata
import json


@dataclass
class SatelliteMetadata:
    '''
        Class that serves as a method of data version controlling based on
        satellite,containing all time stamps in the tile directory for each
        satellite. Five SatelliteMetadata Objects should be instantiated per
        tile directory (Tile1, Tile2, etc.). There are 5 because one exists
        for each satellite type, and an additional for the ground truth.
        This does not include the np array.
    '''
    satellite_type: str
    bands: List[str]
    times: List[str]
    original_filenames: List[List[str]]


@dataclass
class TileMetadata:
    '''
    A 50x50 pixel area in the satellite image corresponds to a 1x1 pixel area
    in the ground truth label Metadata x_gt, y_gt correspond to pixel in
    ground truth (gt) while file name of subtiles is referencing pixel
    coordinates of parent image.
    '''
    satellites: Dict[str, SatelliteMetadata]
    x_gt: int  # dependent upon the number of pixels in ground truth tile
    # ex: 2 stride_size/tile_size_gt  equals 8 
    y_gt: int
    subtile_size: int  # This is wrt the parent tile (4x4 in gt --> 200x200 in
    # parent) keeping in mind the scaling factor of 800x800 mapping to 16x16
    # which makes the scaling factor between the parent and ground
    # truth 50x50 (parent pixel grouping) for every 1x1 (ground truth)

    parent_tile_id: str  # Tile1, Tile2,... identifies Tile subdirectory

    # in Train folder

    # Utility functions provided for generating JSON files. Do not edit unless
    # you know what you are doing.
    def toJSON(self):
        # multitasking json.dump what default is doing is utilizing a lambda
        # function to convert dataclass to dict, then dict to json.
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def saveJSON(self, file_name: str | os.PathLike):
        with open(file_name, "w", encoding='utf-8') as outfile:
            return json.dump(self, outfile,
                             default=lambda o: o.__dict__,
                             sort_keys=True, indent=4)

    @staticmethod
    def loadJSON(file_name: str | os.PathLike):
        # loading tilemetadata and building satellitemetadata for restitching
        with open(file_name, "r", encoding="utf-8") as infile:
            json_data = infile.read()
        metadata_dict = json.loads(json_data)

        # the toJSON function saves our TileMetadata object as a dictionary
        # including the SatelliteMetadata objects (we would need to pickle
        # in order to preserve user defined types), so we need to access
        # "satellites" key to convert the dictionary of dictionaries to a
        # dictionary back to SatelliteMetadata objects.

        # Create a dictionary for each Tile JSON that contains the parent
        # satellites' metadata
        per_tile_satellites = {}
        for sat_type in metadata_dict["satellites"]:
            per_tile_satellites[sat_type] = SatelliteMetadata(
                **(metadata_dict["satellites"][sat_type])
            )
        metadata_dict["satellites"] = per_tile_satellites
        # Returns a TileMetaData object constructed by unpacking the 
        # metadata_dict (unpacks by using ** op)
        return TileMetadata(**metadata_dict)


class Subtile:
    """
    This class saves the satellite stack and tile metadata for a subtile by
    creating the corresponding directories and saving the files. It also loads
    the files and returns the satellite stack and tile metadata using the
    helper function tile_filename_to_metadata_filename.
    """

    def __init__(
            self,
            satellite_stack: Dict[str, np.ndarray] | None = None,
            tile_metadata: TileMetadata | None = None
    ):
        self.satellite_stack = satellite_stack
        self.tile_metadata = tile_metadata

    def save(self, dir: str | os.PathLike):
        """
        Create directories for ~/data/processed/Train/Tile1/subtiles
        and ~/data/processed/Train/Tile1/metadata
        Do not recreate the directories if they already exist.
        """
        dir = Path(dir)
        (dir / "subtiles").mkdir(parents=True, exist_ok=True)
        (dir / "metadata").mkdir(parents=True, exist_ok=True)

        # Save your stacked and sliced numpy arrays to the 
        # ~/data/processed/Train/<Parent_Tile_ID>/subtiles directory
        # Filenames should follow the format: 
        # <ParentTileDirectoryName>_<column>_<row>.npz
        # (ParentTileDirectoryName refers to Tile1, Tile2, etc.)
        # Hint - You can use np.savez or np.save
        np.savez(
            dir / "subtiles" / f"{self.tile_metadata.parent_tile_id}_{self.tile_metadata.x_gt}_{self.tile_metadata.y_gt}.npz",
            **self.satellite_stack)

        # Save the tile_metadata as a JSON in the ~/data/processed/Train/<Parent_Tile_ID>/metadata
        # using the utility function provided in TileMetaData
        # Use the same format you used to save the subtiles, but save it as a .json file
        self.tile_metadata.saveJSON(
            dir / "metadata" / f"{self.tile_metadata.parent_tile_id}_{self.tile_metadata.x_gt}_{self.tile_metadata.y_gt}.json")

    def tile_filename_to_metadata_filename(self, file_name: str | os.PathLike):
        file_name = Path(file_name)
        name_of_file = file_name.stem
        return file_name.parent.parent / "metadata" / f"{name_of_file}.json"

    def load(self, file_name: str | os.PathLike):
        """
        Load the satellite stack and tile metadata from the directories
        """
        self.satellite_stack = dict(np.load(file_name))
        metadata_file_name = self.tile_filename_to_metadata_filename(file_name)
        self.tile_metadata = TileMetadata.loadJSON(metadata_file_name)
        return self


def metadata_to_tile_metadata(
        metadata_stack: Dict[str, List[Metadata]],
        x_gt: int,
        y_gt: int,
        tile_size_gt: int
) -> TileMetadata:
    """
    Converts a stack of metadata files, i.e., a dictionary which maps the name of the satellite to a list of Metadata objects
    into a TileMetadata object that holds the metadata of a single subtile.

    Input:
        metadata_stack: Dict[str,List[Metadata]]
            Mapping from satellite to List[Metadata]
        x_gt: int
            x coordinate of the subtile
        y_gt: int
            y coordinate of the subtile
        tile_size_gt: int
            resolution of the subtile, keep in mind that tile_size_satellite = 50*tile_size_gt, i.e.,
            1 ground truth pixel maps onto a 50x50 satellite tile.
    Output:
        tile_metadata: TileMetadata
            TileMetadata object with information relating to a tile
    """
    satellite_metadata = {}
    for satellite_type in metadata_stack:
        satellite_metadata[satellite_type] = SatelliteMetadata(
            satellite_type,
            metadata_stack[satellite_type][0].bands,
            [m.time for m in metadata_stack[satellite_type]],
            [m.file_name for m in metadata_stack[satellite_type]]
        )

        parent_tile_id = metadata_stack[satellite_type][0].tile_id

    tile_metadata = TileMetadata(
        satellites=deepcopy(satellite_metadata),
        x_gt=x_gt,
        y_gt=y_gt,
        parent_tile_id=parent_tile_id,
        subtile_size=tile_size_gt
    )
    return tile_metadata


def tile_metadata_to_metadata(
        tile_metadata: TileMetadata
) -> Dict[str, List[Metadata]]:
    """
    Converts a TileMetadata object to a dict that maps satellites to a List[Metadata] object.
    Inverts the operations done by `metadata_to_tile_metadata`, used to aid with using
    past plotting functions which require the old metadata format.

    Input: 
        tile_metadata: TileMetadata
            TileMetadata object with information relating to a tile
    Output:
        metadata_stack: Dict[str,List[Metadata]]
                Mapping from satellite to List[Metadata]
    """
    metadata_stack = {}
    for satellite_type in tile_metadata.satellites:
        metadata = []
        for i, time in enumerate(
                tile_metadata.satellites[satellite_type].times):
            metadata.append(
                Metadata(
                    satellite_type=satellite_type,
                    file_name=tile_metadata.satellites[satellite_type].original_filenames[i],
                    tile_id=tile_metadata.parent_tile_id,
                    bands=tile_metadata.satellites[satellite_type].bands,
                    time=time
                )
            )
        metadata_stack[satellite_type] = metadata
    return metadata_stack


# -------------------Start Here ------------------------------ #
def get_tile_ground_truth(
        gt_parent: np.ndarray,
        x_gt: int,
        y_gt: int,
        size_gt: int | Tuple[int, int]
) -> np.ndarray:
    """
    This function returns the associated slice of the parent ground truth image
    based on size_gt and the x and y coordinates of the top left corner of the
    subimage corresponding to the ground truth image.

    Given a parent tile, coordinates (x_gt, y_gt) and a square size size_gt,
    returns the pixels on gt_parent between (size_gt[0]*x_gt, size_gt[1]*y_gt)
    and (size_gt[0]*(x_gt+1), size_gt[1]*(y_gt+1))

    HINT: Use np.copy() to copy the array rather than returning a slice, if
    you save a sliced array to a file, it will save the whole array, using up
    more disk space and memory when loaded.

    Inputs:
        gt_parent: np.ndarray
            ground truth array
        x_gt: int
            top left index of tile
        y_gt: int
            top right index of tile
        size_gt: int | Tuple[int, int]
            Size of the slice, if a tuple is inputted,
            the first value will be used for the x dimension and the second
            value for the y dimension. If an int is inputted, it will be used
            for both dimensions.
    Outputs:
        gt_child: np.ndarray
            Shape of array is (time, bands, size_gt[0], size_gt[1]), where
            time and bands are 1
    and bands is 1 for ground truth.
    """
    # check if size_gt is an int, if so convert to a tuple
    if type(size_gt) is int:
        size_gt = (size_gt, size_gt)

    # calculate the start and end indices for the slice based
    # on the size_gt and the x and y coordinates
    start_x = int(size_gt[0] * x_gt)
    start_y = int(size_gt[1] * y_gt)
    end_x = int(size_gt[0] * (x_gt + 1))
    end_y = int(size_gt[1] * (y_gt + 1))

    # check to make sure the slice is within the bounds of
    # the ground truth parent image
    if start_x < x_gt or start_y < y_gt \
            or start_x > gt_parent.shape[2] or start_y > gt_parent.shape[3] \
            or end_x < x_gt or end_y < y_gt \
            or end_x > gt_parent.shape[2] or end_y > gt_parent.shape[3]:
        raise ValueError("The slice is out of bounds of the ground truth parent image")

    # return the slice of the ground truth parent image
    return np.copy(gt_parent[:, :, start_x:end_x, start_y:end_y])


def get_tile_satellite(
        sat_parent: np.ndarray,
        x_sat: int,
        y_sat: int,
        size_gt: int | Tuple[int, int],
        scale_factor=50
) -> np.ndarray:
    """
    This function returns the associated slice of the parent satellite image
    based on size_gt and the x and y coordinates of the top left corner of the
    subimage corresponding to the ground truth image.

    Given a parent tile, coordinates (x_sat, y_sat), a square size size_sat,
    and a scale factor, returns the pixels on gt_parent between
    (scale_factor*size_gt[0]*x_sat, scale_factor*size_gt[1]*y_sat)
    and (scale_factor*size_gt[0]*(x_sat+1), scale_factor*size_gt[1]*(y_sat+1))

    HINT: Use np.copy() to copy the array rather than returning a slice, if
    you save a sliced array to a file, it will save the whole array, using up
    more disk space and memory when loaded.
    
    Inputs:
        sat_parent: np.ndarray
            parent satellite image numpy array stack
        x_sat: int
            top left index of tile
        y_sat: int
            top right index of tile
        size_gt: int | Tuple[int, int]
            Size of the slice in terms of ground truth pixels. If a tuple is inputted,
            the first value will be used for the x dimension and the second 
            value for the y dimension. If an int is inputted, it will be used
            for both dimensions.
        scale_factor: int
            Ratio between size of the ground truth image and satellite image.
            For example, in this problem the satellite image is 800x800 and
            the ground truth image is 16x16, thus the scale factor is
            800/16 = 50
    outputs:
        sat_child: np.ndarray
            Shape of array is
            (time, bands, size_gt[0]*scale_factor, size_gt[1]*scale_factor),
            where time and bands are 1
    and bands is 1 for ground truth.
    """
    # scale_factor = satellite.length / ground_truth.length

    # check if size_gt is an int, if so convert to a tuple
    if type(size_gt) is int:
        size_gt = (size_gt, size_gt)

    # calculate the start and end indices for the slice based
    # on the size_gt and the x and y coordinates & scale factor
    start_x = int(size_gt[0] * x_sat * scale_factor)
    start_y = int(size_gt[1] * y_sat * scale_factor)
    end_x = int(size_gt[0] * (x_sat + 1) * scale_factor)
    end_y = int(size_gt[1] * (y_sat + 1) * scale_factor)

    # check to make sure the slice is within the bounds of
    # the satellite parent image
    if start_x < x_sat or start_y < y_sat \
            or start_x > sat_parent.shape[2] or start_y > sat_parent.shape[3] \
            or end_x < x_sat or end_y < y_sat \
            or end_x > sat_parent.shape[2] or end_y > sat_parent.shape[3]:
        raise ValueError("The slice is out of bounds of the ground truth parent image")

    # return the slice of the satellite parent image
    return np.copy(sat_parent[:, :, start_x:end_x, start_y:end_y])


def grid_slice(
        satellite_stack: Dict[str, np.ndarray],
        metadata_stack: Dict[str, List[Metadata]],
        tile_size_gt: int) -> List[Subtile]:
    """
    Slices a satellite image into non-overlapping squares. The cut
    is done as a grid, with the following coordinate system:

    -------------------------
    | (0,0) | (1,0) | (2,0) | } tile_size_gt
    -------------------------
    | (0,1) | (1,1) | (2,1) | ...
    -------------------------
    | (0,2) | (1,2) | (2,2) |
    -------------------------
               .     (---v----)
               .     tile_size_gt
               .
    where each square is of size tile_size_gt for the ground truth images,
    and 50*tile_size_gt for the satellite_images.

    The resulting transformation is saved in a Subtile object.

    The metadata_stack may be converted into a TileMetadata object using the
    metadata_to_tile_metadata function.

    tile_size_gt should be square and with respect to the ground truth image.
    """

    # Fetch a ground truth stack and choose a satellite type
    # to retrieve the scale factor
    gt_stack = satellite_stack['gt']

    # Check if the ground truth stack is divisible by the tile size
    if gt_stack.shape[2] % tile_size_gt != 0 or gt_stack.shape[3] % tile_size_gt != 0:
        raise ValueError("The ground truth stack is not divisible by the tile size")

    # Calculate the number of tiles needed in the x dimension
    num_tiles_x = int(gt_stack.shape[2] / tile_size_gt)

    # Calculate the number of tiles needed in the y dimension
    num_tiles_y = int(gt_stack.shape[3] / tile_size_gt)

    # Provide to students as they have the metadata_to_tile_metadata function
    satellite_metadata = {}
    for satellite_type in satellite_stack:
        satellite_metadata[satellite_type] = SatelliteMetadata(
            satellite_type,
            metadata_stack[satellite_type][0].bands,
            [m.time for m in metadata_stack[satellite_type]],
            [m.file_name for m in metadata_stack[satellite_type]]
        )

        parent_tile_id = metadata_stack[satellite_type][0].tile_id

    # instantiate subtiles list
    subtiles = []

    # iterate on every x and y subtile
    for x in range(num_tiles_x):
        for y in range(num_tiles_y):
            # create tile
            # satellite_tiles = []
            satellite_tiles = {}

            # for every satellite type
            for sat_type in satellite_stack.keys():
                slice = None

                # gt is a special case
                # get the current slice for ground truth using get_tile_ground_truth
                if sat_type == 'gt':
                    slice = get_tile_ground_truth(gt_stack, x, y, tile_size_gt)
                # otherwise use the get_tile_satellite function
                else:
                    scale_factor = satellite_stack[sat_type].shape[2] / gt_stack.shape[2]
                    slice = get_tile_satellite(satellite_stack[sat_type], x, y, tile_size_gt, scale_factor)

                # satellite_tiles.append(slice)
                satellite_tiles[sat_type] = slice

            # create the associated TileMetadata
            tile_metadata = metadata_to_tile_metadata(metadata_stack, x, y, tile_size_gt)

            # instantiate Subtile with satellite_tiles and tile_metadata
            subtile = Subtile(satellite_tiles, tile_metadata)  ### QUESTIONABLE PROLLY WRONG (first param)

            # append to subtile list
            subtiles.append(subtile)

    # return the list of Subtile objects
    return subtiles


def restitch(
        dir: str | os.PathLike,
        satellite_type: str,
        tile_id: str,
        range_x: Tuple[int, int],
        range_y: Tuple[int, int]
) -> Tuple[np.ndarray, List[List[TileMetadata]]]:
    """
    Given a directory of processed subtiles, a tile_id and a satellite_type,
    this function will retrieve the tiles between (range_x[0],range_y[0])
    and (range_x[1],range_y[1]) in order to stitch them together to their
    original image.

    Input:
        dir: str | os.PathLike
            Directory where the subtiles are saved
        satellite_type: str
            Satellite type that will be stitched
        tile_id: str
            Tile id that will be stitched
        range_x: Tuple[int, int]
            Range of tiles that will be stitched on width dimension [0,5)]
        range_y: Tuple[int, int]
            Range of tiles that will be stitched on height dimension

    Output:
        stitched_image: np.ndarray
            Stitched image, of shape (time, bands, width, height)
        satellite_metadata_from_subtile: List[List[TileMetadata]]
    """
    subtiles = []
    metadata = []

    # Iterate over the specified ranges
    for x in range(range_x[0], range_x[1]):
        row_subtiles = []
        row_metadata = []
        for y in range(range_y[0], range_y[1]):
            # Construct the filename for the subtile
            filename = f"{tile_id}_{x}_{y}.npz"
            # Load the subtile
            subtile = Subtile()
            subtile.load(dir/filename)
            # Add the subtile and its metadata to the lists
            row_subtiles.append(subtile.satellite_stack[satellite_type])
            row_metadata.append(subtile.tile_metadata)
        # Stack the subtiles in the row along the y-axis
        row_subtiles = np.concatenate(row_subtiles, axis=3)
        # Add the row of subtiles and metadata to the lists
        subtiles.append(row_subtiles)
        metadata.append(row_metadata)
    # Stack the rows of subtiles along the x-axis
    stitched_image = np.concatenate(subtiles, axis=2)

    return stitched_image, metadata

""" Augmentations Implemented as Callable Classes."""
import cv2
import numpy as np
import torch
import random
from typing import Dict


def apply_per_band(img, transform):
    """
    Helpful function to allow you to more easily implement
    transformations that are applied to each band separately.
    Not necessary to use, but can be helpful.
    """
    result = np.zeros_like(img)
    for band in range(img.shape[0]):
        transformed_band = transform(img[band].copy())
        result[band] = transformed_band

    return result


class Blur(object):
    """
        Blurs each band separately using cv2.blur

        Parameters:
            kernel: Size of the blurring kernel
            in both x and y dimensions, used
            as the input of cv.blur

        This operation is only done to the X input array.
    """

    def __init__(self, kernel=3):
        self.kernel = kernel
        pass

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
            Performs the blur transformation.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """

        def blur_transform(x):
            return cv2.blur(x, (self.kernel, self.kernel))
       
        new_X = apply_per_band(sample["X"], blur_transform)
        

        transformed = {"X": new_X, "y": sample["y"]}

        return transformed

        #  dimensions of img: (t, bands, tile_height, tile_width)


class AddNoise(object):
    """
        Adds random gaussian noise using np.random.normal.

        Parameters:
            mean: float
                Mean of the gaussian noise
            std_lim: float
                Maximum value of the standard deviation
    """

    def __init__(self, mean=0, std_lim=0.):
        self.mean = mean
        self.std_lim = std_lim

    def __call__(self, sample):
        """
            Performs the add noise transformation.
            A random standard deviation is first calculated using
            random.uniform to be between 0 and self.std_lim

            Random noise is then added to each pixel with
            mean self.mean and the standard deviation
            that was just calculated

            The resulting value is then clipped using
            numpy's clip function to be values between
            0 and 1.

            This operation is only done to the X array.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        std = random.uniform(0, self.std_lim)
        old_X = sample["X"]
        new_X = []
        if(len(sample["X"].shape)==4):
            tiles, band, w, h = old_X.shape
            for t in range(tiles):
                arr = []
                for b in range(band):
                    # for every pixel
                    noise = np.random.normal(self.mean, std, (w, h))
                    added_noise = old_X[t, b, :, :] + noise
                    arr.append(np.stack(added_noise))

                new_X.append(np.stack(arr))
        else:
            band, w, h = old_X.shape
            arr = []
            for b in range(band):
                # for every pixel
                noise = np.random.normal(self.mean, std, (w, h))
                added_noise = old_X[ b, :, :] + noise
                arr.append(np.stack(added_noise))

            new_X.append(np.stack(arr))
        

        new_X = np.stack(new_X)

        transformed = {"X": new_X, "y": sample["y"]}

        return transformed


class RandomVFlip(object):
    """
        Randomly flips all bands vertically in an image with probability p.

        Parameters:
            p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.prob = p
        self.flip = False

    def __call__(self, sample):
        """
            Performs the random flip transformation using cv.flip.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        rand = random.random()
        if rand <= self.prob:
            self.flip = True

        if not self.flip:
            return sample

        def VFlip_transform(x):
            return cv2.flip(x, 0)
        
        new_X = apply_per_band(sample["X"], VFlip_transform)
        new_y = apply_per_band(sample["y"], VFlip_transform)

        transformed = {"X": new_X, "y": new_y}

        return transformed


class RandomHFlip(object):
    """
        Randomly flips all bands horizontally in an image with probability p.

        Parameters:
            p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.prob = p
        self.flip = False

    def __call__(self, sample):
        """
            Performs the random flip transformation using cv.flip.

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        rand = random.random()
        if rand <= self.prob:
            self.flip = True

        if not self.flip:
            return sample

        def HFlip_transform(x):
            return cv2.flip(x, 1)

        new_X = apply_per_band(sample["X"], HFlip_transform)
        new_y = apply_per_band(sample["y"], HFlip_transform)

        transformed = {"X": new_X, "y": new_y}

        return transformed


class ToTensor(object):
    """
        Converts numpy.array to torch.tensor
    """

    def __call__(self, sample):
        """
            Transforms all numpy arrays to tensors

            Input:
                sample: Dict[str, np.ndarray]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)

            Output:
                transformed: Dict[str, torch.Tensor]
                    Has two keys, 'X' and 'y'.
                    Each of them has shape (bands, width, height)
        """
        new_X = torch.from_numpy(sample["X"])
        new_y = torch.from_numpy(sample["y"])

        transformed = {"X": new_X, "y": new_y}

        return transformed

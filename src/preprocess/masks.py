"""Mask noise: small random changes to the object outline of a training query image.

The demo gets its masks from SAM, and SAM masks are less clean than the dataset's own masks: their
edges are a few pixels off, and thin parts or small areas can be missing. Training queries with a
slightly changed outline teach the heads not to rely on an exact outline.

The training uses the prepared query images (224 × 224, background already grey), so the noise
works on the object pixels of such an image:

- erosion: the outline moves inwards by a few pixels (1 to 3 pixels of 224; a stronger erosion
  removes thin parts, for example chair legs, completely);
- holes: a few small discs inside the object become background.

Both only remove object pixels. Growing the mask would change nothing, because the pixels outside the
object are already grey; the SAM query images bring the errors that leak into the real background.

The functions return a new mask and never an empty one: if the noise would remove every object
pixel, the original mask is returned.
"""

import numpy as np
from scipy import ndimage


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Move the outline of the mask inwards by `pixels` (0 = no change)."""
    if pixels <= 0:
        return mask.copy()
    return ndimage.binary_erosion(mask, iterations=pixels)


def punch_holes(mask: np.ndarray, rng: np.random.Generator, count: int, radius: int) -> np.ndarray:
    """Set `count` discs of `radius` pixels, centred on random object pixels, to background."""
    result = mask.copy()
    object_pixels = np.argwhere(mask)
    if count <= 0 or radius <= 0 or len(object_pixels) == 0:
        return result
    rows, cols = np.ogrid[: mask.shape[0], : mask.shape[1]]
    for row, col in object_pixels[rng.integers(0, len(object_pixels), size=count)]:
        result[(rows - row) ** 2 + (cols - col) ** 2 <= radius**2] = False
    return result


def mask_noise(mask: np.ndarray, rng: np.random.Generator, noise_cfg: dict) -> np.ndarray:
    """Return a noisy copy of the mask (or the mask itself, with probability 1 − p).

    Args:
        mask: bool (H, W), True = object.
        rng: The random number generator of the data loader worker.
        noise_cfg: `train.augment.mask_noise`:
            p            probability to change the mask at all;
            erode        [min, max] erosion as a share of the longer image side;
            holes_max    largest number of holes (0 to holes_max, at random);
            hole_radius  [min, max] radius of a hole as a share of the longer image side.
    """
    if rng.random() >= noise_cfg["p"]:
        return mask
    side = max(mask.shape)
    noisy = erode(mask, int(round(rng.uniform(*noise_cfg["erode"]) * side)))
    holes = int(rng.integers(0, noise_cfg["holes_max"] + 1))
    noisy = punch_holes(noisy, rng, holes, int(round(rng.uniform(*noise_cfg["hole_radius"]) * side)))
    return noisy if noisy.any() else mask

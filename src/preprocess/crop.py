"""The one crop function of the project.

Queries (photos) and renders must look alike in framing, size, and background, so that the
model compares the objects and not the way the images were made. Thus every image goes
through `crop_to_square`, with the same settings:

    1. Background: replace every pixel outside the object mask with a neutral grey.
    2. Bounding box: find the smallest rectangle that contains every object pixel.
    3. Margin (optional): make the box larger by the same number of pixels on all four sides.
       The number is `margin` times the longer side of the box. Example: a box of 400 x 300
       pixels with margin 0.05 grows by 0.05 x 400 = 20 pixels on each side, to 440 x 340.
       With margin 0.0 (the project setting) the box stays tight around the object.
    4. Cut: keep only the pixels inside the box.
    5. Square: add grey rows or columns to the shorter side, half on each side, so that the
       image becomes a square with the object in the centre.
    6. Resize the square to the output size (for example 224 x 224) with antialiasing.

The result: the longer side of the object fills the frame (when the margin is 0), and the
background is one uniform grey.

The function needs only an image and a True/False mask of the same size. It does not matter
where the mask comes from:

- Pix3D photos: the mask files of the dataset.
- ABO photos: masks from the segmentation step (Grounding DINO finds a box, SAM 2.1 makes the mask).
- Renders: the alpha channel of the renderer.
- Demo photos: a SAM 2.1 mask from the box that the user draws.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class EmptyMaskError(ValueError):
    """The mask has no object pixels, so there is nothing to crop."""


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tightest box (x0, y0, x1, y1) around the True pixels of a mask.

    x1 and y1 are exclusive (one past the last object pixel), as in Python slices.

    Raises:
        EmptyMaskError: If the mask has no True pixel.
    """
    rows = np.flatnonzero(mask.any(axis=1))  # indices of rows that contain object pixels
    cols = np.flatnonzero(mask.any(axis=0))  # indices of columns that contain object pixels
    if rows.size == 0:
        raise EmptyMaskError("the mask contains no object pixels")
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop_to_square(
    image: np.ndarray,
    mask: np.ndarray,
    background_rgb: tuple[int, int, int] = (128, 128, 128),
    margin: float = 0.0,
    output_size: int = 224,
) -> np.ndarray:
    """Mask, crop, pad to a square, and resize one image (see the module description).

    Args:
        image: uint8 array (H, W, 3), RGB.
        mask: bool array (H, W); True = object.
        background_rgb: Colour of the background and of the padding.
        margin: Extra border around the object. The border width in pixels is `margin` times the
            longer side of the object's bounding box (0.05 on a 400-pixel side = 20 pixels on each
            side). 0.0 gives a tight crop. The border is filled with the background colour.
        output_size: Side length of the square result, in pixels.

    Returns:
        uint8 array (output_size, output_size, 3), RGB.

    Raises:
        EmptyMaskError: If the mask has no object pixels.
        ValueError: If the image and the mask do not have the same height and width.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image {image.shape[:2]} and mask {mask.shape} must have the same size")
    background = np.asarray(background_rgb, dtype=np.uint8)

    # 1. Background to grey. mask[..., None] has shape (H, W, 1), so it selects whole RGB pixels.
    masked = np.where(mask[..., None], image, background)

    # 2. Bounding box: the smallest rectangle that contains every object pixel. x0 and y0 are the
    # first object column and row; x1 and y1 are one past the last ones, as in Python slices.
    x0, y0, x1, y1 = mask_bbox(mask)

    # 3. Margin: grow the box by `border` pixels on all four sides. `border` is the margin times the
    # longer side of the box (margin 0.05 and a 400 x 300 box: border = 20, new box 440 x 340).
    # The grown box can extend past the image edges; step 5 fills those parts with grey.
    border = int(round(margin * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = x0 - border, y0 - border, x1 + border, y1 + border

    # 4. Cut: keep the pixels that are inside both the box and the image. The cut can be smaller
    # than the box when the box extends past an image edge (only with a margin larger than 0).
    height, width = mask.shape
    cut = masked[max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)]

    # 5. Square: pad the cut with grey to a square whose side is the longer side of the box.
    # The shorter side gets half of the missing rows (or columns) on each side, so the object is in
    # the centre. Example: a 400 x 300 box (width x height) gives a 400 x 400 square with 50 grey
    # rows above and 50 below. The second term in pad_top and pad_left adds back the box rows and
    # columns that were outside the image in step 4, so that the object keeps its place in the box.
    side = max(x1 - x0, y1 - y0)
    pad_top = (side - (y1 - y0)) // 2 + (max(y0, 0) - y0)
    pad_left = (side - (x1 - x0)) // 2 + (max(x0, 0) - x0)
    pad_bottom = side - cut.shape[0] - pad_top
    pad_right = side - cut.shape[1] - pad_left
    tensor = torch.from_numpy(np.ascontiguousarray(cut)).permute(2, 0, 1).float() / 255.0  # (3, h, w), 0..1
    channels = [
        F.pad(tensor[c : c + 1], (pad_left, pad_right, pad_top, pad_bottom), value=float(background[c]) / 255.0)
        for c in range(3)  # each channel is padded with its own background value
    ]
    square = torch.cat(channels, dim=0)  # (3, side, side)

    # 6. Resize with antialiasing, then back to uint8. Rounding keeps flat areas at their exact value.
    resized = TF.resize(square, [output_size, output_size], antialias=True)
    return (resized.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).numpy()


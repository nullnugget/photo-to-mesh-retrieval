"""The one crop function of the project.

Queries (photos) and renders must look alike in framing, size, and background, so that the
model compares the objects and not the way the images were made. Thus every image goes
through `crop_to_square`, with the same settings:

    1. Replace every pixel outside the object mask with a neutral grey.
    2. Find the tightest box around the mask pixels.
    3. Optionally enlarge the box by a margin (a fraction of its longer side).
    4. Cut the image to the box.
    5. Pad the cut to a square with the same grey, with the object in the centre.
    6. Resize the square to the output size (for example 224 x 224) with antialiasing.

The result: the longer side of the object fills the frame (when the margin is 0), and the
background is one uniform grey. Photos use their annotated or predicted mask; renders use the
alpha channel of the renderer as their mask.
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
        margin: Extra border around the object, as a fraction of the longer side of its box.
            0.0 gives a tight crop. The border is filled with the background colour.
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

    # 2. and 3. The tightest box, enlarged by the margin. The box may now extend past the image edges;
    # step 5 fills such parts with the background colour.
    x0, y0, x1, y1 = mask_bbox(mask)
    border = int(round(margin * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = x0 - border, y0 - border, x1 + border, y1 + border

    # 4. Cut the part of the box that lies inside the image.
    height, width = mask.shape
    cut = masked[max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)]

    # 5. Pad to a square. The square side is the longer side of the box. The padding puts the box
    # in the centre and also restores any part of the box that was outside the image.
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


if __name__ == "__main__":
    # Stand-alone check (from the repository root: python -m src.preprocess.crop).
    # It crops three Pix3D photos and saves each photo, its mask, and its crop to test_outputs/crop/.
    from PIL import Image

    from src import paths
    from src.data.pix3d import load_annotations, photo_records, read_photo_and_mask

    out_dir = paths.ensure_dir(paths.TEST_OUTPUTS_DIR / "crop")
    records = {r.query_id: r for r in photo_records(load_annotations())}
    for query_id in ["pix3d/chair/0001", "pix3d/bookcase/0001", "pix3d/table/0046"]:
        record = records[query_id]
        image, mask = read_photo_and_mask(record.photo_path, record.mask_path)
        crop = crop_to_square(image, mask)
        name = query_id.replace("/", "_")
        Image.fromarray(image).save(out_dir / f"{name}_1_photo.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(out_dir / f"{name}_2_mask.png")
        Image.fromarray(crop).save(out_dir / f"{name}_3_crop.png")
        print(f"{query_id}: photo {image.shape} -> crop {crop.shape}")
    print(f"saved to {out_dir.relative_to(paths.ROOT)}")

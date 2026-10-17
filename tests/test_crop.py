"""Tests for src/preprocess/crop.py, on small synthetic images (no dataset needed)."""

import numpy as np
import pytest

from src.preprocess.crop import EmptyMaskError, crop_to_square, mask_bbox

GREY = (128, 128, 128)
RED = (255, 0, 0)


def _image_with_rectangle(height, width, top, left, bottom, right):
    """A noisy background with a red rectangle; the mask marks the rectangle."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)  # background noise
    image[top:bottom, left:right] = RED
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return image, mask


def _is_red(pixel):
    return pixel[0] > 200 and pixel[1] < 60 and pixel[2] < 60


def test_mask_bbox_is_tight_and_exclusive():
    _, mask = _image_with_rectangle(50, 80, top=10, left=20, bottom=30, right=60)
    assert mask_bbox(mask) == (20, 10, 60, 30)


def test_output_size_and_type():
    image, mask = _image_with_rectangle(120, 200, 30, 40, 90, 180)
    crop = crop_to_square(image, mask, output_size=224)
    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8


@pytest.mark.parametrize("shape", ["wide", "tall"])
def test_object_fills_the_longer_side(shape):
    # A wide object must touch the left and right edges; a tall object the top and bottom edges.
    if shape == "wide":
        image, mask = _image_with_rectangle(200, 300, 80, 20, 120, 280)
        crop = crop_to_square(image, mask)
        assert _is_red(crop[112, 1]) and _is_red(crop[112, -2])
    else:
        image, mask = _image_with_rectangle(300, 200, 20, 80, 280, 120)
        crop = crop_to_square(image, mask)
        assert _is_red(crop[1, 112]) and _is_red(crop[-2, 112])


def test_background_and_padding_are_exactly_grey():
    # A wide object leaves grey padding above and below it; the background noise must be gone.
    image, mask = _image_with_rectangle(200, 300, 80, 20, 120, 280)
    crop = crop_to_square(image, mask, background_rgb=GREY)
    top_band = crop[:40]  # far from the object, so resizing cannot blend in red
    assert (top_band == np.array(GREY, dtype=np.uint8)).all()


def test_object_is_centred():
    image, mask = _image_with_rectangle(200, 300, 80, 20, 120, 280)
    crop = crop_to_square(image, mask)
    red_rows = [y for y in range(224) if _is_red(crop[y, 112])]
    assert abs((red_rows[0] + red_rows[-1]) / 2 - 111.5) <= 1.5


def test_margin_leaves_a_grey_border():
    image, mask = _image_with_rectangle(200, 300, 80, 20, 120, 280)
    crop = crop_to_square(image, mask, margin=0.1)
    assert not _is_red(crop[112, 2]) and (crop[112, 2] == np.array(GREY, dtype=np.uint8)).all()


def test_object_at_the_image_edge_with_margin():
    # The enlarged box extends past the image; the missing part must be filled with grey.
    image, mask = _image_with_rectangle(100, 100, 0, 0, 50, 50)
    crop = crop_to_square(image, mask, margin=0.2)
    assert crop.shape == (224, 224, 3)
    assert (crop[2, 2] == np.array(GREY, dtype=np.uint8)).all()


def test_empty_mask_is_an_error():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(EmptyMaskError):
        crop_to_square(image, np.zeros((10, 10), dtype=bool))


def test_size_mismatch_is_an_error():
    with pytest.raises(ValueError):
        crop_to_square(np.zeros((10, 10, 3), dtype=np.uint8), np.ones((5, 5), dtype=bool))

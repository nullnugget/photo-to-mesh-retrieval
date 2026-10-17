"""Tests for src/preprocess/queries.py, on small synthetic files (no dataset needed)."""

import numpy as np
import pytest
from PIL import Image

from src.preprocess.queries import DONE, FAILED, SKIPPED, MaskMismatchError, QueryJob, make_query_images, read_photo_and_mask


def _write_pair(folder, name, photo_size=(60, 40), mask_size=(60, 40), empty=False):
    """Write a random photo and a mask with a white rectangle (or an empty mask)."""
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 256, size=(photo_size[1], photo_size[0], 3), dtype=np.uint8)
    mask = np.zeros((mask_size[1], mask_size[0]), dtype=np.uint8)
    if not empty:
        mask[10:30, 15:45] = 255
    Image.fromarray(photo).save(folder / f"{name}.png")
    Image.fromarray(mask).save(folder / f"{name}_mask.png")
    return str(folder / f"{name}.png"), str(folder / f"{name}_mask.png")


def _job(folder, name, photo, mask, overwrite=False):
    return QueryJob(name, photo, mask, str(folder / "out" / f"{name}.png"), (128, 128, 128), 0.0, 32, overwrite)


def test_mask_of_another_size_raises(tmp_path):
    photo, mask = _write_pair(tmp_path, "a", mask_size=(50, 40))
    with pytest.raises(MaskMismatchError):
        read_photo_and_mask(photo, mask)


@pytest.mark.parametrize("workers", [1, 2])
def test_jobs_give_done_failed_and_skipped(tmp_path, workers):
    good = _job(tmp_path, "good", *_write_pair(tmp_path, "good"))
    wrong_size = _job(tmp_path, "wrong", *_write_pair(tmp_path, "wrong", mask_size=(50, 40)))
    empty = _job(tmp_path, "empty", *_write_pair(tmp_path, "empty", empty=True))
    results = make_query_images([good, wrong_size, empty], workers=workers)
    assert [r["status"] for r in results] == [DONE, FAILED, FAILED]
    assert "MaskMismatchError" in results[1]["message"] and "EmptyMaskError" in results[2]["message"]
    with Image.open(good.output_path) as query:
        assert query.size == (32, 32)
    assert make_query_images([good], workers=workers)[0]["status"] == SKIPPED  # the file exists now

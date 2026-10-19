"""Tests for src/preprocess/masks.py, src/data/augment.py, and src/data/datasets.py (CPU only)."""

import numpy as np
import pytest
import torch
from PIL import Image

from src import paths
from src.config import load_config
from src.data import datasets
from src.data.augment import QueryAugment
from src.preprocess.masks import erode, mask_noise, punch_holes

GREY = (128, 128, 128)
CFG = load_config(["train.yaml"])["train"]


def _query_image(size=224):
    """A grey square with a coloured object (a rectangle with a stripe) in the middle."""
    image = np.full((size, size, 3), 128, dtype=np.uint8)
    image[40:184, 60:164] = (200, 60, 30)
    image[100:120, 60:164] = (20, 140, 220)
    return image


def _background_is_grey(image, object_mask):
    return bool((image[~object_mask] == np.array(GREY, dtype=np.uint8)).all())


def test_erosion_and_holes_only_remove_object_pixels():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    eroded = erode(mask, 2)
    assert eroded.sum() == 26 * 26 and not (eroded & ~mask).any()
    holed = punch_holes(mask, np.random.default_rng(0), count=2, radius=3)
    assert holed.sum() < mask.sum() and not (holed & ~mask).any()
    assert (erode(mask, 0) == mask).all()


def test_mask_noise_never_returns_an_empty_mask_and_respects_p():
    tiny = np.zeros((20, 20), dtype=bool)
    tiny[9:11, 9:11] = True
    cfg = {"p": 1.0, "erode": [0.5, 0.5], "holes_max": 0, "hole_radius": [0.0, 0.0]}
    assert mask_noise(tiny, np.random.default_rng(0), cfg) is tiny  # erosion would remove everything
    assert mask_noise(tiny, np.random.default_rng(0), {**cfg, "p": 0.0}) is tiny


def test_augmented_images_have_the_right_size_and_a_grey_border():
    augment = QueryAugment(CFG["augment"], GREY, 224)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = augment(_query_image(), rng)
        assert out.shape == (224, 224, 3) and out.dtype == np.uint8
        assert (out[0, 0] == GREY).all() and (out[-1, -1] == GREY).all()  # the corners stay background


def test_rotation_and_zoom_keep_the_background_exactly_grey():
    augment = QueryAugment(CFG["augment"], GREY, 224)
    empty = np.full((224, 224, 3), 128, dtype=np.uint8)
    for seed in range(10):
        assert (augment._geometry(empty, np.random.default_rng(seed)) == 128).all()  # no lines along the turned frame
        out = augment._geometry(_query_image(), np.random.default_rng(seed))
        assert (out[:30] == 128).all() and (out[-30:] == 128).all()  # far from the object: exactly grey


def test_the_same_seed_gives_the_same_changes():
    augment = QueryAugment(CFG["augment"], GREY, 224)
    first = augment(_query_image(), np.random.default_rng(7))
    second = augment(_query_image(), np.random.default_rng(7))
    third = augment(_query_image(), np.random.default_rng(8))
    assert (first == second).all() and not (first == third).all()


def test_colour_changes_touch_only_the_object():
    augment = QueryAugment(CFG["augment"], GREY, 224)
    image = _query_image()
    mask = augment._object(image)
    for seed in range(10):
        out = augment._colour(image, mask, np.random.default_rng(seed))
        assert _background_is_grey(out, mask)


def test_the_occluding_patch_stays_on_the_object():
    augment = QueryAugment({**CFG["augment"], "erasing_p": 1.0}, GREY, 224)
    image = _query_image()
    mask = augment._object(image)
    for seed in range(10):
        out = augment._occlude(image, np.random.default_rng(seed))
        assert _background_is_grey(out, mask)


def test_mask_noise_crops_tight_again():
    cfg = {**CFG["augment"], "mask_noise": {"enabled": True, "p": 1.0, "erode": [0.03, 0.03], "holes_max": 0, "hole_radius": [0.0, 0.0]},
           "color_jitter": {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0}, "grayscale_p": 0.0,
           "rotation_deg": 0, "zoom_out_min": 1.0, "erasing_p": 0.0}
    out = QueryAugment(cfg, GREY, 224)(_query_image(), np.random.default_rng(0))
    rows = np.flatnonzero((out != GREY).any(axis=2).any(axis=1))
    assert rows[0] == 0 and rows[-1] == 223  # the eroded object fills the frame again (its longer side is vertical)


def test_query_image_files_by_kind():
    pix3d = {"dataset": "pix3d", "category": "chair", "query_path": "data/derived/queries/pix3d/chair/0001.png"}
    abo = {"dataset": "abo", "category": "lamp", "query_path": "data/derived/queries/abo/lamp/X.png"}
    assert datasets.query_image_file(pix3d, "reference") == paths.ROOT / pix3d["query_path"]
    assert datasets.query_image_file(pix3d, "sam") == paths.MASKS_DIR / "pix3d_sam_queries" / "chair" / "0001.png"
    assert datasets.query_image_file(abo, "sam") == paths.ROOT / abo["query_path"]  # ABO: the catalog image is the SAM one
    with pytest.raises(ValueError):
        datasets.query_image_file(pix3d, "box")


def test_training_photos_keep_the_images_that_exist(tmp_path, monkeypatch):
    files = {name: tmp_path / f"{name}.png" for name in ("q1_ref", "q1_sam", "q2_ref")}
    for file in files.values():
        Image.fromarray(_query_image(32)).save(file)
    rows = [{"query_id": q, "dataset": "pix3d", "category": "chair", "mesh_id": "m"} for q in ("q1", "q2", "q3")]
    monkeypatch.setattr(datasets, "read_table", lambda path: rows)
    monkeypatch.setattr(datasets, "load_split", lambda path: {"parts": {"train": ["q1", "q2", "q3"]}})
    monkeypatch.setattr(datasets, "query_image_file", lambda row, kind: files.get(f"{row['query_id']}_{kind[:3]}", tmp_path / "none.png"))
    photos, report = datasets.training_photos(["pix3d"], "seen_meshes", "train", {"pix3d": ["reference", "sam"]})
    assert [len(p.images) for p in photos] == [2, 1]
    assert report["left_out"] == ["q3"] and report["images"] == {"pix3d_reference": 2, "pix3d_sam": 1}


def test_the_dataset_returns_changed_images_with_their_numbers(tmp_path):
    file = tmp_path / "q.png"
    Image.fromarray(_query_image()).save(file)
    photo = datasets.TrainingPhoto("pix3d/chair/0001", "pix3d", "chair", "pix3d/chair/M", (str(file),))
    dataset = datasets.TrainingQueries([photo], QueryAugment(CFG["augment"], GREY, 224), {"pix3d/chair/M": 3}, {"chair": 1}, seed=0)
    item = dataset[0]
    assert item["image"].shape == (224, 224, 3) and item["image"].dtype == torch.uint8
    assert item["mesh"] == 3 and item["category"] == 1
    assert not torch.equal(dataset[0]["image"], item["image"])  # a new change at every use

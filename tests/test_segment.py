"""Tests for src/preprocess/segment.py and the cleaning rules of src/pipeline/segment_photos.py.

The model tests run on the CPU (a few seconds each), so they do not need the GPU. They use SAM 2.1
small, which behaves like the active SAM 2.1 large at a quarter of the time.
"""

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.pipeline.segment_photos import KEPT, REMOVED, cleaning_verdict
from src.preprocess import segment


def test_prompt_text_uses_the_table_or_the_class_name():
    assert segment.prompt_text("tv_stand", {"tv_stand": "tv stand"}) == "tv stand."
    assert segment.prompt_text("ceiling_light") == "ceiling light."
    assert segment.prompt_text("Sofa.") == "sofa."


def test_edges_touched_counts_each_edge_once():
    assert segment.edges_touched((0, 0, 100, 50), width=100, height=100) == 3  # left, top, right
    assert segment.edges_touched((10, 10, 90, 90), width=100, height=100) == 0
    assert segment.edges_touched((1.5, 20, 50, 98.5), width=100, height=100) == 2  # within 2 pixels


def test_mask_iou_with_and_without_ignored_pixels():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:, :6] = True  # 60 pixels
    b[:, 4:] = True  # 60 pixels, 20 shared
    assert segment.mask_iou(a, b) == pytest.approx(20 / 100)
    ignore = np.zeros((10, 10), bool)
    ignore[:, 6:] = True  # ignore the part only b covers
    assert segment.mask_iou(a, b, ignore) == pytest.approx(20 / 60)
    assert segment.mask_iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 1.0


def test_edge_band_holds_only_pixels_near_the_edge():
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    band = segment.edge_band(mask, width=2)
    assert band[10, 20] and band[9, 20] and band[11, 20]  # on, just outside, and just inside the edge
    assert not band[20, 20] and not band[2, 2]  # the centre and far outside
    assert not segment.edge_band(mask, width=0).any()
    assert segment.edge_band_width(np.zeros((480, 640), bool), 0.005) == 3


def test_mask_file_has_the_pix3d_format(tmp_path):
    mask = np.zeros((5, 7), bool)
    mask[1:3, 2:5] = True
    path = segment.save_mask(mask, tmp_path / "sub" / "m.png")
    with Image.open(path) as image:
        assert image.mode == "L" and image.size == (7, 5)
        assert set(np.unique(np.asarray(image)).tolist()) == {0, 255}  # black background, white object
    assert (segment.read_mask(path) == mask).all()
    assert not list(tmp_path.rglob("*.part"))  # no temporary file is left


def test_box_of_mask_is_one_past_the_object():
    mask = np.zeros((20, 30), bool)
    mask[5:9, 10:25] = True
    assert segment.box_of_mask(mask) == (10.0, 5.0, 25.0, 9.0)


def test_cleaning_rules():
    rules = {"max_edges_touched": 2, "close_up_box_share": 0.6, "min_box_side": 0.3, "min_box_pixels": 112,
             "min_mask_area": 0.01, "max_mask_area": 0.9}
    w, h = 1000, 800
    assert cleaning_verdict((100, 100, 600, 700), w, h, 0.3, rules)[0] == KEPT  # inside the photo, large enough
    assert cleaning_verdict((0, 0, 1000, 700), w, h, 0.5, rules)[:2] == (REMOVED, "close_up")  # 3 edges
    assert cleaning_verdict((0, 50, 800, 750), w, h, 0.5, rules)[:2] == (REMOVED, "close_up")  # 1 edge, box 0.70
    assert cleaning_verdict((600, 200, 1000, 700), w, h, 0.2, rules)[0] == KEPT  # 1 edge, box 0.25: cut by the frame
    assert cleaning_verdict((400, 300, 600, 500), w, h, 0.03, rules)[:2] == (REMOVED, "box_too_small")  # 200 < 300 pixels
    assert cleaning_verdict((450, 20, 550, 780), w, h, 0.03, rules)[0] == KEPT  # thin and tall: a floor lamp
    assert cleaning_verdict((10, 10, 90, 100), 120, 120, 0.3, rules)[:2] == (REMOVED, "box_too_small")  # under 112 pixels
    assert cleaning_verdict((100, 100, 600, 700), w, h, 0.005, rules)[:2] == (REMOVED, "mask_too_small")
    assert cleaning_verdict((100, 100, 600, 700), w, h, 0.95, rules)[:2] == (REMOVED, "mask_too_large")


@pytest.mark.weights
def test_sam_box_prompt_finds_a_plain_block():
    # A synthetic photo: a dark grey block on a white background. The expected mask is the block
    # (white = object, black = background, like the Pix3D masks).
    photo = np.full((240, 320, 3), 255, np.uint8)
    photo[60:180, 80:240] = (40, 40, 40)
    block = np.zeros((240, 320), bool)
    block[60:180, 80:240] = True
    segmenter = segment.Segmenter(load_config()["segment"]["segmenter_small"], device="cpu")
    result = segmenter.mask_from_box(photo, (74, 54, 246, 186))  # a box a little larger than the block
    assert result.mask.shape == (240, 320) and result.mask.dtype == bool
    assert segment.mask_iou(result.mask, block) > 0.95
    clicked = segmenter.mask_from_points(photo, [(160, 120)])  # one click in the centre of the block
    assert segment.mask_iou(clicked.mask, block) > 0.9


@pytest.mark.weights
def test_unknown_box_mask_choice_is_an_error():
    with pytest.raises(ValueError):
        segment.Segmenter({**load_config()["segment"]["segmenter_small"], "box_mask": "smallest"}, device="cpu")


@pytest.mark.weights
@pytest.mark.data
def test_detector_finds_the_chair_of_a_pix3d_photo():
    photo = np.asarray(Image.open("data/raw/pix3d/img/chair/0001.png").convert("RGB"))
    reference = segment.read_mask("data/raw/pix3d/mask/chair/0001.png")
    detector = segment.Detector(load_config()["segment"]["detector"], device="cpu")
    detections = detector.detect(photo, "chair.")
    assert detections and detections[0].label == "chair"
    found = np.zeros_like(reference)
    x0, y0, x1, y1 = (round(v) for v in detections[0].box)
    found[y0:y1, x0:x1] = True
    expected = np.zeros_like(reference)
    rx0, ry0, rx1, ry1 = (int(v) for v in segment.box_of_mask(reference))
    expected[ry0:ry1, rx0:rx1] = True
    assert segment.mask_iou(found, expected) > 0.8  # the detected box matches the box of the reference mask


@pytest.mark.weights
def test_one_encoding_gives_the_same_masks_as_separate_calls():
    photo = np.full((240, 320, 3), 255, np.uint8)
    photo[60:180, 80:240] = (40, 40, 40)
    segmenter = segment.Segmenter(load_config()["segment"]["segmenter_small"], device="cpu")
    box = (74, 54, 246, 186)
    both = segmenter.masks_for_each_choice(photo, box)
    for choice in segment.BOX_MASK_CHOICES:
        alone = segmenter.mask_from_box(photo, box, choice=choice)
        assert (both[choice].mask == alone.mask).all()
        assert both[choice].score == pytest.approx(alone.score, abs=1e-3)  # float rounding differs slightly


def test_grow_mask_adds_a_ring_and_stays_in_the_box():
    mask = np.zeros((20, 20), bool)
    mask[8:12, 8:12] = True
    grown = segment.grow_mask(mask, 2)
    assert grown[6:14, 6:14].all() and grown.sum() == 64  # 4 x 4 grown by 2 on each side = 8 x 8
    clipped = segment.grow_mask(mask, 2, box=(8, 8, 12, 14))  # the box allows growth only downwards
    assert clipped.sum() == 4 * 6 and clipped[8:14, 8:12].all()
    assert (segment.grow_mask(mask, 0) == mask).all()


def test_box_mask_covers_the_box_only():
    inside = segment.box_mask((10, 20), (2, 3, 7, 9))
    assert inside.sum() == 5 * 6 and inside[3:9, 2:7].all()

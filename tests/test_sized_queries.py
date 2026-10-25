"""Tests for the file names of the images of another size (src/pipeline/sized_queries.py; no images needed)."""

import pytest

from src import paths
from src.pipeline import sized_queries
from src.pipeline.extract_features import catalog_query_file, sam_query_file

ROW = {"query_id": "pix3d/chair/0001", "dataset": "pix3d", "query_path": "data/derived/queries/pix3d/chair/0001.png"}


def test_the_main_size_keeps_the_main_files():
    assert catalog_query_file(ROW) == paths.ROOT / ROW["query_path"]
    assert sam_query_file(paths.MASKS_DIR, ROW) == paths.MASKS_DIR / "pix3d_sam_queries" / "chair" / "0001.png"


def test_another_size_has_the_same_names_in_its_own_folders():
    assert catalog_query_file(ROW, "_448") == paths.DERIVED_DIR / "queries_448" / "pix3d" / "chair" / "0001.png"
    assert sam_query_file(paths.MASKS_DIR, ROW, "_448") == paths.MASKS_DIR / "pix3d_sam_queries_448" / "chair" / "0001.png"
    assert sized_queries.target_file("pix3d", "queries_reference", ROW, "_448", False) == catalog_query_file(ROW, "_448")
    assert sized_queries.target_file("pix3d", "queries_sam", ROW, "_448", False) == sam_query_file(paths.MASKS_DIR, ROW, "_448")


def test_the_mini_run_writes_below_test_outputs():
    target = sized_queries.target_file("pix3d", "queries_sam", ROW, "_448", True)
    assert target == paths.TEST_OUTPUTS_DIR / "sized_queries_mini" / "masks" / "pix3d_sam_queries_448" / "chair" / "0001.png"


def test_the_main_size_is_refused():
    cfg = {"data": {"image_size": 224}, "backbone": {"name": "dinov3_vitb16"}}
    with pytest.raises(ValueError):
        sized_queries.run(cfg, 224, ["pix3d"])

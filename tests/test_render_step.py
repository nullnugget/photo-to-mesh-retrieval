"""Tests for the safety parts of src/pipeline/render_meshes.py (no rendering needed)."""

import csv

import numpy as np
import pytest

from src import paths
from src.pipeline import render_meshes


def test_changed_settings_stop_the_step(tmp_path):
    render_meshes._check_settings({"lighting": "L2_dark"}, tmp_path, overwrite=False)  # first run: record saved
    render_meshes._check_settings({"lighting": "L2_dark"}, tmp_path, overwrite=False)  # same settings: fine
    with pytest.raises(render_meshes.RenderSettingsChanged):
        render_meshes._check_settings({"lighting": "L0"}, tmp_path, overwrite=False)
    render_meshes._check_settings({"lighting": "L0"}, tmp_path, overwrite=True)  # overwrite accepts and records


def test_views_are_written_completely_or_not_at_all(tmp_path):
    folder = tmp_path / "pix3d" / "chair" / "X"
    (tmp_path / "pix3d" / "chair" / "X.part").mkdir(parents=True)  # left over from an interrupted run
    render_meshes._save_views(np.zeros((25, 8, 8, 3), dtype=np.uint8), folder)
    assert len(list(folder.glob("view_*.png"))) == 25 and not (tmp_path / "pix3d" / "chair" / "X.part").exists()


def test_skipped_meshes_keep_their_earlier_results(tmp_path):
    path = tmp_path / "results.csv"
    row = {"views_touching_border": 0, "seconds": 1.0, "message": ""}
    render_meshes._merge_results(path, [{"mesh_id": "a", "status": "done", "overflow_retries": 2, **row}])
    merged = render_meshes._merge_results(path, [{"mesh_id": "a", "status": "skipped", "overflow_retries": 0, **row},
                                                 {"mesh_id": "b", "status": "done", "overflow_retries": 0, **row}])
    assert [(r["mesh_id"], r["status"], str(r["overflow_retries"])) for r in merged] == [("a", "done", "2"), ("b", "done", "0")]
    with open(path, newline="") as f:
        assert len(list(csv.DictReader(f))) == 2
    merged = render_meshes._merge_results(path, [{"mesh_id": "c", "status": "done", "overflow_retries": 0, **row}])
    assert [r["mesh_id"] for r in merged] == ["a", "b", "c"]  # a run on other meshes keeps the earlier rows


def test_renders_of_another_size_go_to_their_own_folder():
    assert render_meshes.output_root(False) == paths.RENDERS_DIR
    assert render_meshes.output_root(False, paths.size_suffix(448, 224)) == paths.DERIVED_DIR / "renders_448"
    assert render_meshes.output_root(True, "_448").name == "renders_448"
    assert paths.size_suffix(224, 224) == ""

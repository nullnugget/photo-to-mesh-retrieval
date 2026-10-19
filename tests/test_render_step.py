"""Tests for the safety parts of src/pipeline/render_meshes.py (no rendering needed)."""

import csv

import numpy as np
import pytest

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

"""Tests for src/runs.py. All files go into pytest's temporary folder."""

import datetime

import pytest
import torch

from src import runs
from src.config import load_config


def test_run_name_format_and_collision_suffix(scratch_run_dirs):
    date = datetime.date(2026, 10, 22)
    name = runs.make_run_name("Pix3D", "dinov3_vitb16", "L2 test", date=date, **scratch_run_dirs)
    assert name == "2026-10-22_pix3d_dinov3_vitb16_l2_test"
    (scratch_run_dirs["checkpoints_dir"] / name).mkdir(parents=True)
    assert runs.make_run_name("Pix3D", "dinov3_vitb16", "L2 test", date=date, **scratch_run_dirs) == name + "_2"


def test_create_run_makes_both_folders_and_saves_the_config(scratch_run_dirs):
    run = runs.create_run(load_config(), "pix3d", "dinov3_vitb16", "check", **scratch_run_dirs)
    assert run["run_dir"].parent == scratch_run_dirs["runs_dir"]
    assert run["checkpoint_dir"].parent == scratch_run_dirs["checkpoints_dir"]
    assert run["run_dir"].name == run["checkpoint_dir"].name == run["run_name"]
    assert (run["run_dir"] / "config.yaml").exists()


def test_registry_header_once_and_unknown_columns_rejected(tmp_path):
    registry = tmp_path / "registry.csv"
    runs.append_registry({"run_name": "a", "test_R@1": 1.0}, registry=registry)
    runs.append_registry({"run_name": "b"}, registry=registry)
    rows = runs.read_registry(registry)
    assert [r["run_name"] for r in rows] == ["a", "b"]
    assert list(rows[0]) == runs.REGISTRY_COLUMNS
    with pytest.raises(KeyError):
        runs.append_registry({"run_name": "c", "not_a_column": 1}, registry=registry)


def _checkpoint(tmp_path):
    features = tmp_path / "features.npz"
    features.write_bytes(b"fake features")
    meta = runs.checkpoint_metadata("run_x", {"k": 1}, "dinov3_vitb16", features, 256, 4, 0.5)
    heads = {k: {"weight": torch.ones(3)} for k in runs.HEAD_KEYS}
    return runs.save_checkpoint(tmp_path / "embedding_heads_best.pt", heads, meta), meta


def test_checkpoint_holds_all_heads_and_describes_itself(tmp_path):
    path, meta = _checkpoint(tmp_path)
    ckpt = runs.load_checkpoint(path)
    assert set(ckpt) == set(runs.HEAD_KEYS) | {"meta"}
    for key in ("run_name", "config", "git_commit", "backbone", "feature_file_sha256", "embedding_dim", "epoch", "val_r1"):
        assert key in ckpt["meta"]
    assert torch.equal(ckpt["view_projector"]["weight"], torch.ones(3))


def test_checkpoint_needs_all_three_heads(tmp_path):
    with pytest.raises(KeyError):
        runs.save_checkpoint(tmp_path / "x.pt", {"query_projector": {}}, {})


@pytest.mark.parametrize(
    "expected",
    [{"backbone": "dinov2_vitb14_reg"}, {"embedding_dim": 128}, {"feature_file_sha256": "0" * 64}],
)
def test_checkpoint_mismatch_is_detected(tmp_path, expected):
    path, _ = _checkpoint(tmp_path)
    with pytest.raises(runs.CheckpointMismatch):
        runs.load_checkpoint(path, expected)


def test_matching_checkpoint_loads(tmp_path):
    path, meta = _checkpoint(tmp_path)
    runs.load_checkpoint(path, {k: meta[k] for k in runs.COMPATIBILITY_KEYS})


def test_seed_makes_random_numbers_repeat():
    runs.set_seed(123)
    a = torch.rand(3)
    runs.set_seed(123)
    assert torch.equal(a, torch.rand(3))

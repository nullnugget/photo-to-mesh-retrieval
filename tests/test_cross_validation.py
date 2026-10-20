"""Tests for src/train/cross_validation.py (synthetic fold runs; no training)."""

import json

import numpy as np
import pytest
import yaml

from src.train import cross_validation
from src.train.trainer import RECORDED_MIX_WEIGHTS

CONFIG = {"train": {"scoring": {"mix_weight": 0.5}}, "backbone": {"feature_recipe": "cls"}}


def _fold_run(root, k, ranks, query_ids, mesh_ids, config=CONFIG):
    """A run folder as `06_train.py` writes it: metrics.json, config.yaml, val_ranks.npz (ranks: epochs, W, Q)."""
    run_dir = root / f"2026-10-20_toy_bb_unseen_meshes_fold{k}_cv"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"run_name": run_dir.name, "split": f"unseen_meshes_fold{k}"}))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    np.savez(run_dir / "val_ranks.npz", mix_weights=np.array(RECORDED_MIX_WEIGHTS), toy_sam__ranks=np.asarray(ranks, dtype=np.int16),
             toy_sam__query_ids=np.array(query_ids), toy_sam__mesh_ids=np.array(mesh_ids), toy_sam__categories=np.array(["chair"] * len(query_ids)))
    return run_dir


def _ranks(epochs, per_weight):
    """Ranks (epochs, W, Q) where every epoch has the same ranks; per_weight: {weight: ranks of the photos}."""
    rows = [per_weight.get(w, per_weight[0.0]) for w in RECORDED_MIX_WEIGHTS]
    return np.stack([rows] * epochs)


def test_the_pool_puts_the_folds_together_and_chooses_one_epoch(tmp_path):
    # Fold 1: two photos of mesh a; fold 2: two photos of mesh b. Zero-shot gets only mesh a right.
    r1 = _ranks(3, {0.0: [1, 1], 0.5: [1, 1]})
    r2 = _ranks(3, {0.0: [2, 2], 0.5: [2, 2]})
    r2[1, RECORDED_MIX_WEIGHTS.index(0.5)] = [1, 1]  # the mix gets mesh b right only at epoch 2
    dirs = [_fold_run(tmp_path, 1, r1, ["q1", "q2"], ["a", "a"]), _fold_run(tmp_path, 2, r2, ["q3", "q4"], ["b", "b"])]
    result = cross_validation.pool(dirs, "unseen_meshes")
    assert result["selected_epoch"] == 2 and result["selection_value"] == 1.0 and result["selection_value_last_epoch"] == 0.5
    row = next(r for r in result["results"] if r["mix_weight"] == 0.5)
    assert row["photos"] == 4 and row["meshes"] == 2 and row["R@1"] == 1.0 and row["vs_zero_shot"] == 0.5
    assert sorted(result["pooled"]["toy_sam"]["query_ids"].tolist()) == ["q1", "q2", "q3", "q4"]
    assert result["pooled"]["toy_sam"]["fold"].tolist() == [1, 1, 2, 2]


def test_the_pool_refuses_runs_that_are_not_one_cross_validation(tmp_path):
    r = _ranks(2, {0.0: [1]})
    a = _fold_run(tmp_path / "a", 1, r, ["q1"], ["a"])
    b = _fold_run(tmp_path / "a", 2, r, ["q1"], ["a"])  # the same photo in two folds
    with pytest.raises(cross_validation.FoldMismatch, match="more than one fold"):
        cross_validation.pool([a, b], "unseen_meshes")
    c = _fold_run(tmp_path / "c", 2, r, ["q2"], ["b"], config={**CONFIG, "train": {"scoring": {"mix_weight": 1.0}}})
    with pytest.raises(cross_validation.FoldMismatch, match="other settings"):
        cross_validation.pool([a, c], "unseen_meshes")
    with pytest.raises(cross_validation.FoldMismatch, match="not the folds"):
        cross_validation.pool([a], "seen_meshes")


def test_two_cross_validations_compare_on_the_same_photos(tmp_path):
    def pooled(name, hits_at_half):
        ranks = np.array([[1, 1, 2, 2]] * len(RECORDED_MIX_WEIGHTS), dtype=np.int16)
        ranks[RECORDED_MIX_WEIGHTS.index(0.5)] = [1 if h else 2 for h in hits_at_half]
        p = {"toy_sam": {"query_ids": np.array(["q1", "q2", "q3", "q4"]), "mesh_ids": np.array(["a", "a", "b", "b"]),
                         "categories": np.array(["chair"] * 4), "fold": np.array([1, 1, 2, 2]), "ranks": ranks}}
        return cross_validation.write_pooled_ranks(p, list(RECORDED_MIX_WEIGHTS), tmp_path / f"{name}.npz")
    a, b = pooled("a", [1, 1, 0, 0]), pooled("b", [1, 1, 1, 1])
    result = cross_validation.compare(a, b, 0.5, 0.5)
    assert result["toy_sam"]["difference"] == 0.5 and result["all"]["meshes"] == 2
    assert cross_validation.compare(a, a, 0.0, 0.5)["all"]["difference"] == 0.0  # a's mix equals its zero-shot here


def test_a_stopped_fold_run_is_not_found(tmp_path):
    r = _ranks(2, {0.0: [1]})
    finished = _fold_run(tmp_path, 1, r, ["q1"], ["a"])
    stopped = tmp_path / (finished.name + "_2")
    stopped.mkdir()
    (stopped / "val_ranks.npz").write_bytes((finished / "val_ranks.npz").read_bytes())  # no metrics.json: stopped
    assert cross_validation.find_fold_runs("toy", "bb", "unseen_meshes", 1, "cv", tmp_path) == [finished]

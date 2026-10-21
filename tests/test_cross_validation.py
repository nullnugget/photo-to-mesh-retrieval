"""Tests for src/train/cross_validation.py (synthetic fold runs; no training)."""

import json

import numpy as np
import pytest
import yaml

from src.train import cross_validation
from src.train.trainer import RECORDED_MIX_WEIGHTS

CONFIG = {"train": {"scoring": {"mix_weight": 0.5}}, "backbone": {"feature_recipe": "cls"}}


def _fold_run(root, k, ranks, query_ids, mesh_ids, config=CONFIG, extra_sets=None):
    """A run folder as `06_train.py` writes it: metrics.json, config.yaml, val_ranks.npz (ranks: epochs, W, Q).

    extra_sets: {query set: (ranks, query_ids, mesh_ids)} recorded next to "toy_sam".
    """
    run_dir = root / f"2026-10-20_toy_bb_unseen_meshes_fold{k}_cv"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"run_name": run_dir.name, "split": f"unseen_meshes_fold{k}", "best_epoch": 1,
                                                      "mix_weight": config["train"]["scoring"]["mix_weight"]}))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    arrays = {}
    for name, (r, q, m) in {"toy_sam": (ranks, query_ids, mesh_ids), **(extra_sets or {})}.items():
        arrays.update({f"{name}__ranks": np.asarray(r, dtype=np.int16), f"{name}__query_ids": np.array(q),
                       f"{name}__mesh_ids": np.array(m), f"{name}__categories": np.array(["chair"] * len(q))})
    np.savez(run_dir / "val_ranks.npz", mix_weights=np.array(RECORDED_MIX_WEIGHTS), **arrays)
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


def test_the_epoch_rule_last_takes_the_last_epoch(tmp_path):
    r = _ranks(3, {0.0: [2], 0.5: [2]})
    r[1, RECORDED_MIX_WEIGHTS.index(0.5)] = [1]  # epoch 2 is the best
    run = _fold_run(tmp_path, 1, r, ["q1"], ["a"])
    best = cross_validation.pool([run], "unseen_meshes")  # no train.final_epoch in the settings: "best"
    assert best["epoch_rule"] == "best" and best["selected_epoch"] == 2
    last = cross_validation.pool([run], "unseen_meshes", "last")
    assert last["selected_epoch"] == 3 and last["best_epoch_of_curve"] == 2 and last["selection_value"] == 0.0
    run2 = _fold_run(tmp_path / "b", 1, r, ["q1"], ["a"], config={**CONFIG, "train": {**CONFIG["train"], "final_epoch": "last"}})
    assert cross_validation.pool([run2], "unseen_meshes")["selected_epoch"] == 3  # the rule of the settings


def test_the_pool_takes_a_query_set_from_the_folds_that_have_it(tmp_path):
    # A joint fold by category: fold 2 has no photo of the second dataset ("abo_sam").
    r = _ranks(1, {0.0: [1]})
    a = _fold_run(tmp_path, 1, r, ["q1"], ["a"], extra_sets={"abo_sam": (r, ["x1"], ["m1"])})
    b = _fold_run(tmp_path, 2, r, ["q2"], ["b"])
    result = cross_validation.pool([a, b], "unseen_meshes")
    assert result["pooled"]["toy_sam"]["query_ids"].tolist() == ["q1", "q2"]
    assert result["pooled"]["abo_sam"]["query_ids"].tolist() == ["x1"] and result["pooled"]["abo_sam"]["fold"].tolist() == [1]


def test_a_joint_and_a_single_training_compare_on_their_shared_query_sets(tmp_path):
    # a: one dataset; b: the same val photos plus a second dataset's set. Only "toy_sam" is compared.
    ra = _ranks(2, {0.0: [1, 2], 0.5: [1, 2]})
    rb = _ranks(2, {0.0: [1, 2], 0.5: [1, 1]})
    rb[0, RECORDED_MIX_WEIGHTS.index(0.5)] = [2, 2]  # b is worse at epoch 1, better at the last epoch
    a = _fold_run(tmp_path / "a", 1, ra, ["q1", "q2"], ["m1", "m2"])
    b = _fold_run(tmp_path / "b", 1, rb, ["q1", "q2"], ["m1", "m2"], extra_sets={"abo_sam": (ra, ["x1", "x2"], ["n1", "n2"])})
    (weights_a, sets_a), (weights_b, sets_b) = cross_validation.run_ranks(a, "last"), cross_validation.run_ranks(b, "last")
    assert sets_a["toy_sam"]["ranks"].shape == (len(RECORDED_MIX_WEIGHTS), 2)
    result = cross_validation.compare_sets(sets_a, sets_b, weights_a, weights_b, 0.5, 0.5)
    assert sorted(result) == ["all", "toy_sam"] and result["toy_sam"]["difference"] == 0.5
    first = cross_validation.compare_sets(sets_a, cross_validation.run_ranks(b, 1)[1], weights_a, weights_b, 0.5, 0.5)
    assert first["toy_sam"]["difference"] == -0.5
    with pytest.raises(cross_validation.FoldMismatch, match="no shared"):
        cross_validation.compare_sets(sets_a, {"abo_sam": sets_b["abo_sam"]}, weights_a, weights_b, 0.5, 0.5)

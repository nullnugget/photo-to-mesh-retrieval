"""Cross-validation by mesh: the folds, and the pooled validation of their k trainings.

Two kinds of folds: by mesh ("unseen_meshes": k folds of whole meshes) and by category
("unseen_categories": one fold per non-test category, leave-one-category-out). Everything after
the folds (training, records, pool, comparison) is the same for both. With category folds the
intervals over meshes understate the uncertainty, because the categories (few) differ more than
the meshes; the results per category (`per_category.csv`) are the main result there.

Why: the val part of a split by mesh holds few meshes (Pix3D unseen_meshes: 39). The photos of one
mesh succeed or fail together, so a val result on 39 meshes depends mostly on which meshes were
drawn. On Pix3D, the val part showed +6 R@1 for the score mix over zero-shot; the test part
showed 0. A cross-validation validates on every non-test mesh once (Pix3D: 315 meshes, 8,020
photos), so a setting is judged on all of them.

How (the folds: `src/data/splits.py`, `mesh_folds`):

    non-test meshes ──► k folds (by mesh, stratified by category); the test part is not touched
    training 1: train on folds 2..k, validate on fold 1     (split "unseen_meshes_fold1")
    ...
    training k: train on folds 1..k−1, validate on fold k   (split "unseen_meshes_fold<k>")
    pool:       the val photos of all k trainings together = every non-test photo once

Each training records the rank of the correct mesh of every val photo, for every epoch and for
the mix weights of `trainer.RECORDED_MIX_WEIGHTS` (`runs/<run>/val_ranks.npz`). The pool reads
these records; nothing is calculated again.

The pooled result:

- the curve: pooled R@1 per epoch (each epoch over all k trainings), for each mix weight;
- one epoch count for all folds: the epoch with the best mean pooled R@1 over the val query sets,
  at the mix weight of the training. One fixed count (not the best epoch of each fold) is what a
  final training on all non-test meshes can use, and it chooses 1 value from the curve, not k;
- at that epoch, for each val query set and mix weight: R@1, R@5, R@10, the 95 % interval of R@1
  over meshes, the paired difference to zero-shot (mix weight 0) over meshes, and R@1 per category.

Two cross-validations of different settings have the same val photos, so `compare` gives their
paired difference over meshes.
"""

import json
from pathlib import Path

import numpy as np
import yaml

from src import paths
from src.data.catalog import read_table
from src.data.splits import category_folds, fold_name, load_split, mesh_folds, save_split
from src.retrieval import metrics

POOLED_RANKS = "pooled_ranks.npz"


class FoldMismatch(RuntimeError):
    """The fold trainings do not form one cross-validation (missing folds, other settings, overlaps)."""


FOLD_KINDS = ("mesh", "category")


def _split_rows(dataset: str, split: str, splits_dir: Path) -> tuple[dict, list[dict]]:
    """The split and the catalog rows of its photos."""
    base = load_split(paths.split_file(dataset, split, splits_dir))
    wanted = set(base["parts"]["train"]) | set(base["parts"]["val"]) | set(base["parts"]["test"])
    return base, [r for r in read_table(paths.QUERIES_CSV) if r["dataset"] == dataset and r["query_id"] in wanted]


def category_fold_count(dataset: str, split: str, splits_dir: Path = paths.SPLITS_DIR) -> int:
    """The number of non-test categories of a split (the folds of a cross-validation by category)."""
    base, rows = _split_rows(dataset, split, splits_dir)
    test = set(base["parts"]["test"])
    return len({r["category"] for r in rows if r["query_id"] not in test})


def ensure_folds(dataset: str, split: str, folds: int, seed: int, kind: str = "mesh",
                 splits_dir: Path = paths.SPLITS_DIR) -> list[str]:
    """Make the fold split files of a split if they do not exist; return the fold names.

    Args:
        kind: "mesh" (k folds of whole meshes, `mesh_folds`) or "category" (one fold per
            non-test category, `category_folds`; `folds` must equal the number of categories).

    The folds depend only on the split, the catalog, and the seed, so existing files are checked
    against a new calculation; a difference (for example after a new catalog) is an error.

    Raises:
        FoldMismatch: If an existing fold file differs from the calculation, or if the number of
            category folds is not `folds`.
    """
    if kind not in FOLD_KINDS:
        raise ValueError(f"kind must be one of {FOLD_KINDS}, got {kind!r}")
    base, rows = _split_rows(dataset, split, splits_dir)
    made = mesh_folds(base, rows, folds, seed) if kind == "mesh" else category_folds(base, rows, seed)
    if len(made) != folds:
        raise FoldMismatch(f"{dataset} {split}: {len(made)} folds, not {folds}")
    for split_dict in made:
        path = paths.split_file(dataset, split_dict["name"], splits_dir)
        if not path.exists():
            save_split(split_dict, path)
        elif load_split(path) != split_dict:
            raise FoldMismatch(f"{path} differs from the folds of the current catalog and seed; delete the fold files to make them again")
    return [s["name"] for s in made]


def _read_run(run_dir: Path) -> dict:
    record = json.loads((run_dir / "metrics.json").read_text())
    with np.load(run_dir / "val_ranks.npz", allow_pickle=False) as data:
        record["ranks"] = {key: data[key] for key in data.files}
    record["config"] = yaml.safe_load((run_dir / "config.yaml").read_text())
    return record


def find_fold_runs(label: str, backbone: str, split: str, folds: int, tag: str, runs_dir: Path = paths.RUNS_DIR) -> list[Path]:
    """Return the run folder of each fold training: `<date>_<label>_<backbone>_<split>_fold<k>_<tag>`.

    Raises:
        FoldMismatch: If a fold has no run or more than one run with this tag.
    """
    found = []
    for k in range(1, folds + 1):
        pattern = f"*_{label}_{backbone}_{fold_name(split, k)}_{tag}"
        # A finished run has metrics.json (written at its end); a stopped run has none and is skipped.
        matches = [p for p in sorted(runs_dir.glob(pattern)) + sorted(runs_dir.glob(pattern + "_[0-9]*"))
                   if (p / "metrics.json").exists() and (p / "val_ranks.npz").exists()]
        if len(matches) != 1:
            raise FoldMismatch(f"fold {k}: {len(matches)} finished runs match {pattern!r} in {runs_dir}"
                               + (f": {[m.name for m in matches]}" if matches else ""))
        found.append(matches[0])
    return found


def pool(run_dirs: list[Path], split: str) -> dict:
    """Pool the val records of the k fold trainings of one cross-validation.

    Args:
        run_dirs: One run folder per fold.
        split: The split that was divided, for example "unseen_meshes".

    Returns:
        {"folds", "epochs", "mix_weights", "mix_weight", "selected_epoch", "curve", "results",
         "per_category", "runs", "config", "pooled"}; "pooled" holds the per-photo arrays at the
        selected epoch (see `write_pooled_ranks`).

    Raises:
        FoldMismatch: If the runs are not the folds 1..k of `split` exactly once, if their settings
            differ, if their val photos overlap, or if they have different numbers of epochs.
    """
    records = [_read_run(Path(d)) for d in run_dirs]
    folds = len(records)
    names = sorted(r["split"] for r in records)
    if names != sorted(fold_name(split, k) for k in range(1, folds + 1)):
        raise FoldMismatch(f"the runs are the splits {names}, not the folds 1..{folds} of {split!r}")
    for r in records[1:]:
        if r["config"] != records[0]["config"]:
            raise FoldMismatch(f"{r['run_name']} has other settings than {records[0]['run_name']}; a cross-validation needs the same settings in every fold")
    epochs = {r["ranks"][next(k for k in r["ranks"] if k.endswith("__ranks"))].shape[0] for r in records}
    if len(epochs) != 1:
        raise FoldMismatch(f"the fold trainings have different numbers of epochs: {sorted(epochs)}")
    epochs = epochs.pop()
    weights = records[0]["ranks"]["mix_weights"].tolist()
    mix_weight = float(records[0]["config"]["train"]["scoring"]["mix_weight"])
    if mix_weight not in weights:
        raise FoldMismatch(f"the mix weight {mix_weight} of the training is not among the recorded weights {weights}")

    sets = sorted({key.split("__")[0] for r in records for key in r["ranks"] if key.endswith("__ranks")})
    pooled = {}
    for name in sets:
        parts = {field: np.concatenate([r["ranks"][f"{name}__{field}"] for r in records], axis=-1)
                 for field in ("query_ids", "mesh_ids", "categories")}
        if len(set(parts["query_ids"].tolist())) != len(parts["query_ids"]):
            raise FoldMismatch(f"{name}: a val photo is in more than one fold")
        parts["ranks"] = np.concatenate([r["ranks"][f"{name}__ranks"] for r in records], axis=-1)  # (epochs, W, N)
        parts["fold"] = np.concatenate([np.full(r["ranks"][f"{name}__ranks"].shape[-1], int(r["split"].rpartition("_fold")[2]))
                                        for r in records])
        pooled[name] = parts

    # The curve, and one epoch count for all folds.
    w_index = weights.index(mix_weight)
    curve = [{"epoch": e + 1, "mix_weight": w, "query_set": name,
              "R@1": float((pooled[name]["ranks"][e, i] <= 1).mean()), "R@5": float((pooled[name]["ranks"][e, i] <= 5).mean())}
             for e in range(epochs) for i, w in enumerate(weights) for name in sets]
    selection = [np.mean([(pooled[name]["ranks"][e, w_index] <= 1).mean() for name in sets]) for e in range(epochs)]
    selected = int(np.argmax(selection))  # the first epoch with the highest value

    results, per_category = [], []
    zero = weights.index(0.0)
    for name in sets:
        p = pooled[name]
        ranks = p["ranks"][selected]  # (W, N)
        for i, w in enumerate(weights):
            hits = (ranks[i] <= 1).astype(float)
            low, high = metrics.bootstrap_interval(hits, groups=p["mesh_ids"])
            versus = metrics.paired_difference((ranks[zero] <= 1).astype(float), hits, groups=p["mesh_ids"])
            results.append({"query_set": name, "mix_weight": w, "photos": len(hits), "meshes": len(set(p["mesh_ids"].tolist())),
                            **{f"R@{k}": float((ranks[i] <= k).mean()) for k in (1, 5, 10)},
                            "R@1_low_meshes": low, "R@1_high_meshes": high,
                            "vs_zero_shot": versus["difference"], "vs_zero_shot_low": versus["low"], "vs_zero_shot_high": versus["high"],
                            "R@1_last_epoch": float((p["ranks"][-1, i] <= 1).mean())})
        for category in sorted(set(p["categories"].tolist())):
            member = p["categories"] == category
            per_category.append({"query_set": name, "category": category, "photos": int(member.sum()),
                                 "meshes": len(set(p["mesh_ids"][member].tolist())),
                                 **{f"R@1_mix{w:g}": float((ranks[i][member] <= 1).mean()) for i, w in enumerate(weights)}})
    return {
        "split": split, "folds": folds, "epochs": epochs, "mix_weights": weights, "mix_weight": mix_weight,
        "selected_epoch": selected + 1, "selection_value": float(selection[selected]),
        "selection_value_last_epoch": float(selection[-1]),
        "curve": curve, "results": results, "per_category": per_category,
        "runs": [r["run_name"] for r in records], "config": records[0]["config"],
        "pooled": {name: {**{k: v for k, v in p.items() if k != "ranks"}, "ranks": p["ranks"][selected]} for name, p in pooled.items()},
    }


def write_pooled_ranks(pooled: dict, mix_weights: list[float], path: Path) -> Path:
    """Save the per-photo ranks at the selected epoch (for `compare`)."""
    arrays = {"mix_weights": np.array(mix_weights)}
    for name, p in pooled.items():
        arrays.update({f"{name}__{field}": (value.astype(str) if value.dtype.kind in "US" else value) for field, value in p.items()})
    np.savez(path, **arrays)
    return path


def read_pooled_ranks(path: Path) -> tuple[list[float], dict[str, dict]]:
    """Read `pooled_ranks.npz`: the mix weights and {query set: {"query_ids", "mesh_ids", "categories", "fold", "ranks"}}."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    sets: dict[str, dict] = {}
    for key, value in arrays.items():
        if "__" in key:
            name, field = key.split("__")
            sets.setdefault(name, {})[field] = value
    return arrays["mix_weights"].tolist(), sets


def compare(path_a: Path, path_b: Path, weight_a: float, weight_b: float) -> dict:
    """Paired difference (b − a) of two cross-validations over the same val photos, over meshes.

    Args:
        path_a, path_b: Their `pooled_ranks.npz` files.
        weight_a, weight_b: The mix weight of each side (for example each one's training value).

    Returns:
        {query set or "all": {"photos", "meshes", "R@1_a", "R@1_b", "difference", "low", "high"}};
        "all" puts the photos of every query set together (a photo then counts once per set).

    Raises:
        FoldMismatch: If the two have different val photos.
    """
    (weights_a, a), (weights_b, b) = read_pooled_ranks(path_a), read_pooled_ranks(path_b)
    if sorted(a) != sorted(b):
        raise FoldMismatch(f"different val query sets: {sorted(a)} and {sorted(b)}")
    out, all_a, all_b, all_groups = {}, [], [], []
    for name in sorted(a):
        order_b = {q: i for i, q in enumerate(b[name]["query_ids"].tolist())}
        if set(order_b) != set(a[name]["query_ids"].tolist()):
            raise FoldMismatch(f"{name}: the two cross-validations have different val photos")
        take = np.array([order_b[q] for q in a[name]["query_ids"].tolist()])
        hits_a = (a[name]["ranks"][weights_a.index(weight_a)] <= 1).astype(float)
        hits_b = (b[name]["ranks"][weights_b.index(weight_b)][take] <= 1).astype(float)
        groups = a[name]["mesh_ids"]
        p = metrics.paired_difference(hits_a, hits_b, groups=groups)
        out[name] = {"photos": len(hits_a), "meshes": len(set(groups.tolist())), "R@1_a": hits_a.mean(), "R@1_b": hits_b.mean(),
                     "difference": p["difference"], "low": p["low"], "high": p["high"]}
        all_a.append(hits_a)
        all_b.append(hits_b)
        all_groups.append(groups)
    groups = np.concatenate(all_groups)
    p = metrics.paired_difference(np.concatenate(all_a), np.concatenate(all_b), groups=groups)
    out["all"] = {"photos": len(groups), "meshes": len(set(groups.tolist())), "R@1_a": float(np.concatenate(all_a).mean()),
                  "R@1_b": float(np.concatenate(all_b).mean()), "difference": p["difference"], "low": p["low"], "high": p["high"]}
    return out

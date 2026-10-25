"""The cross-validation step: train one setting on every fold of a split, then pool and record the result.

Steps (the method: `src/train/cross_validation.py`; the folds: `src/data/splits.py`; by mesh for
unseen_meshes, by category for unseen_categories, `data.splits.<split>.cross_validation`):

1. make the fold split files if they do not exist (`data/splits/<dataset>/<split>_fold<k>.json`);
   in a joint training by category, the categories of the first dataset decide the folds of all;
2. one training per fold (`src/pipeline/train_heads.py`), with the same settings and the same tag;
3. when every fold has a finished run: pool the val records at one epoch for all folds
   (`train.final_epoch`), and record the pooled result.

Comparisons (`compare_pools`, `compare_runs`): the paired difference over meshes of two
cross-validations, or of two single trainings, on the val query sets that both have. A Pix3D-only
and a joint (Pix3D + ABO) training compare on the Pix3D val photos.

Outputs (a mini run writes into `test_outputs/train_mini/` instead):
    runs/<date>_<datasets>_<backbone>_<fold run name parts>       one training run per fold (as `06_train.py`)
    runs/<date>_<datasets>_<backbone>_<split>_cv<k>_<tag>/
        config.yaml          the settings (the same in every fold)
        metrics.json         the selected epoch, the pooled results per query set and mix weight, the fold runs
        curve.csv            pooled R@1 and R@5 per epoch, mix weight, and query set
        per_category.csv     pooled R@1 per category at the selected epoch, for every mix weight
        pooled_ranks.npz     the rank of every val photo at the selected epoch (for comparisons)
    runs/registry.csv        one row per val query set (the selected epoch, the mix weight of the training)
"""

import csv
import datetime
import json
import shutil
from pathlib import Path

import torch

from src import paths, runs
from src.data.splits import fold_name
from src.pipeline import train_heads
from src.train import cross_validation


def fold_plan(cfg: dict, split: str, datasets: list[str]) -> tuple[str, int]:
    """The kind and the number of folds of a split.

    `data.splits.<split>.cross_validation`: an integer k = k folds by mesh; "by_category" = one fold
    per non-test category of the first dataset (the other datasets follow its categories).
    """
    setting = cfg["data"]["splits"].get(split, {}).get("cross_validation")
    if setting == "by_category":
        return "category", cross_validation.category_fold_count(datasets[0], split)
    if isinstance(setting, int) and setting > 1:
        return "mesh", setting
    raise ValueError(f"the split {split!r} has no cross_validation setting (an integer k, or by_category)")


def run_folds(cfg: dict, split: str, tag: str, folds: list[int] | None = None, datasets: list[str] | None = None,
              backbone: str | None = None, mini: bool = False, gallery_only: list[str] | None = None) -> dict | None:
    """Train one setting on the given folds; pool and record when every fold has a run with this tag.

    Args:
        cfg: The merged settings (the same for every fold).
        split: The split to divide, for example "unseen_meshes".
        tag: The name of this cross-validation; it finds the fold runs again (required).
        folds: Fold numbers to train now (None = all). A stopped cross-validation continues with the missing folds.
        datasets, backbone, mini, gallery_only: As in `train_heads.run`.

    Returns:
        The pooled record, or None if folds are still missing.
    """
    if not tag:
        raise ValueError("a cross-validation needs a tag (it finds the fold runs by it)")
    datasets = datasets or list(cfg["data"]["datasets"])
    backbone = backbone or cfg["backbone"]["name"]
    kind, count = fold_plan(cfg, split, datasets)
    lead = cross_validation.fold_categories(datasets[0], split) if kind == "category" else None
    for dataset in datasets:
        cross_validation.ensure_folds(dataset, split, count, cfg["data"]["splits"]["seed"], kind,
                                      categories=lead if dataset != datasets[0] else None)
    folds = folds or list(range(1, count + 1))
    for k in folds:
        if not 1 <= k <= count:
            raise ValueError(f"fold {k} does not exist (1 to {count})")
    for k in folds:
        print(f"\n=== fold {k} of {count} ===", flush=True)
        train_heads.run(cfg, split=fold_name(split, k), datasets=datasets, backbone=backbone, tag=tag, mini=mini,
                        gallery_only=gallery_only)
        torch.cuda.empty_cache()
    try:
        return pool_and_record(split, tag, count, datasets, backbone, mini)
    except cross_validation.FoldMismatch as error:
        print(f"\nnot pooled yet: {error}", flush=True)
        return None


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pool_and_record(split: str, tag: str, folds: int, datasets: list[str], backbone: str, mini: bool = False,
                    epoch: str | None = None) -> dict:
    """Pool the fold runs of one cross-validation (found by tag), print the result, and record it.

    `epoch`: "last" or "best" (None = `train.final_epoch` of the trainings).
    """
    out = train_heads.output_dirs(mini)
    label = "_".join(datasets)
    run_dirs = cross_validation.find_fold_runs(label, backbone, split, folds, tag, out["runs"])
    result = cross_validation.pool(run_dirs, split, epoch)

    name = runs.make_run_name(label, backbone, f"{split}_cv{folds}_{tag}", runs_dir=out["runs"], checkpoints_dir=out["checkpoints"])
    run_dir = paths.ensure_dir(out["runs"] / name)
    shutil.copy(run_dirs[0] / "config.yaml", run_dir / "config.yaml")
    _write_csv(result["curve"], run_dir / "curve.csv")
    _write_csv(result["per_category"], run_dir / "per_category.csv")
    cross_validation.write_pooled_ranks(result["pooled"], result["mix_weights"], run_dir / cross_validation.POOLED_RANKS)
    commit, dirty = runs.git_state()
    record = {"run_name": name, "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "git_commit": commit, "git_dirty": dirty, "mini": mini, "datasets": datasets, "backbone": backbone,
              **{k: v for k, v in result.items() if k not in ("pooled", "curve", "per_category", "config")}}
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")

    train_cfg = result["config"]["train"]
    w = result["mix_weight"]
    print(f"\ncross-validation {tag}: {folds} folds of {split}, {result['epochs']} epochs; epoch {result['selected_epoch']} "
          f"({result['epoch_rule']}; mean pooled R@1 {100 * result['selection_value']:.1f}; "
          f"best epoch of the curve {result['best_epoch_of_curve']})", flush=True)
    for r in result["results"]:
        print(f"  {r['query_set']:24s} mix {r['mix_weight']:4g}  photos {r['photos']:5d}  meshes {r['meshes']:3d}  "
              f"R@1 {100 * r['R@1']:5.1f} (meshes {100 * r['R@1_low_meshes']:.1f} to {100 * r['R@1_high_meshes']:.1f})  "
              f"vs zero-shot {100 * r['vs_zero_shot']:+5.1f} ({100 * r['vs_zero_shot_low']:+.1f} to {100 * r['vs_zero_shot_high']:+.1f})  "
              f"R@5 {100 * r['R@5']:5.1f}", flush=True)
        if r["mix_weight"] != w:
            continue
        runs.append_registry({
            "run_name": name, "date": record["date"][:10], "git_commit": commit[:7], "git_dirty": dirty, "mode": "trained",
            "datasets": label, "split": f"{split}_cv{folds}", "photos": "val", "photo_count": r["photos"],
            "query_set": r["query_set"].split("_", 1)[1], "backbone": backbone, "recipe": result["config"]["backbone"]["feature_recipe"],
            "scoring": "combined", "alpha": train_cfg["scoring"]["alpha"], "mix_weight": w,
            "render_config": result["config"]["render"]["rig"], "lighting": result["config"]["render"]["lighting"],
            "temperature": train_cfg["loss"]["temperature"], "category_weight": train_cfg["loss"]["category_weight"],
            "epochs": result["epochs"], "best_epoch": result["selected_epoch"], "val_R@1": round(100 * result["selection_value"], 2),
            "R@1": round(100 * r["R@1"], 2), "R@5": round(100 * r["R@5"], 2), "R@10": round(100 * r["R@10"], 2),
            "notes": f"pooled val of {folds} folds ({r['meshes']} meshes), one epoch count for all folds; vs zero-shot "
                     f"{100 * r['vs_zero_shot']:+.1f} (meshes {100 * r['vs_zero_shot_low']:+.1f} to {100 * r['vs_zero_shot_high']:+.1f})"
                     + ("; mini" if mini else ""),
        }, registry=out["registry"])
    print(f"-> {run_dir.relative_to(paths.ROOT)}", flush=True)
    return record


def find_pool(split: str, tag: str, folds: int, backbone: str, mini: bool = False) -> Path:
    """Return the folder of the pooled result of a cross-validation (the newest one with this tag, any datasets)."""
    runs_dir = train_heads.output_dirs(mini)["runs"]
    pattern = f"*_{backbone}_{split}_cv{folds}_{tag}"
    found = sorted(list(runs_dir.glob(pattern)) + list(runs_dir.glob(pattern + "_[0-9]*")),
                   key=lambda p: p.stat().st_mtime)
    found = [p for p in found if (p / cross_validation.POOLED_RANKS).exists()]
    if not found:
        raise FileNotFoundError(f"no pooled cross-validation {pattern!r} in {runs_dir}")
    return found[-1]


def _pooled_sets(pool_dir: Path, split: str, folds: int, tag: str, backbone: str, mini: bool, epoch: str | None) -> tuple[list[float], dict]:
    """The pooled per-photo ranks of a cross-validation: from its pooled_ranks.npz, or (with `epoch`) pooled again from its fold runs."""
    if epoch is None:
        return cross_validation.read_pooled_ranks(pool_dir / cross_validation.POOLED_RANKS)
    label = "_".join(json.loads((pool_dir / "metrics.json").read_text())["datasets"])
    run_dirs = cross_validation.find_fold_runs(label, backbone, split, folds, tag, train_heads.output_dirs(mini)["runs"])
    result = cross_validation.pool(run_dirs, split, epoch)
    return result["mix_weights"], result["pooled"]


def _print_comparison(title: str, result: dict, only_a: list[str], only_b: list[str]) -> None:
    print(title)
    for name, r in result.items():
        print(f"  {name:24s} photos {r['photos']:5d}  meshes {r['meshes']:3d}  R@1 {100 * r['R@1_a']:5.1f} -> {100 * r['R@1_b']:5.1f}  "
              f"difference {100 * r['difference']:+5.1f} ({100 * r['low']:+.1f} to {100 * r['high']:+.1f})")
    for side, names in (("first", only_a), ("second", only_b)):
        if names:
            print(f"  not compared (only in the {side}): {', '.join(names)}")


def compare_pools(split: str, tag_a: str, tag_b: str, folds: int, backbone: str,
                  weight_a: float | None = None, weight_b: float | None = None, mini: bool = False,
                  epoch: str | None = None) -> dict:
    """Print and return the paired difference (b − a) of two cross-validations, over meshes.

    The mix weight of each side defaults to the mix weight of its training. `epoch` ("last" or
    "best") pools both again from their fold runs at that epoch; None uses each one's recorded pool.
    The two can have different datasets (Pix3D-only and joint): the shared val query sets are compared.
    """
    dirs = [find_pool(split, tag, folds, backbone, mini) for tag in (tag_a, tag_b)]
    trained_w = [json.loads((d / "metrics.json").read_text())["mix_weight"] for d in dirs]
    weight_a = trained_w[0] if weight_a is None else weight_a
    weight_b = trained_w[1] if weight_b is None else weight_b
    (weights_a, a), (weights_b, b) = (_pooled_sets(d, split, folds, tag, backbone, mini, epoch) for d, tag in zip(dirs, (tag_a, tag_b)))
    result = cross_validation.compare_sets(a, b, weights_a, weights_b, weight_a, weight_b)
    _print_comparison(f"{tag_b} (mix {weight_b:g}) − {tag_a} (mix {weight_a:g}), paired over meshes"
                      + (f", epoch: {epoch}" if epoch else ", the recorded pools") + ":",
                      result, sorted(set(a) - set(b)), sorted(set(b) - set(a)))
    return result


def compare_runs(run_a: str, run_b: str, weight_a: float | None = None, weight_b: float | None = None,
                 mini: bool = False, epoch: str = "last") -> dict:
    """Print and return the paired difference (b − a) of two single trainings (by run name), over meshes.

    For example a Pix3D-only and a joint training of seen_meshes, on the Pix3D val photos. The mix
    weight of each side defaults to the mix weight of its training; `epoch` is "last" or "best".
    """
    runs_dir = train_heads.output_dirs(mini)["runs"]
    dirs = [runs_dir / name for name in (run_a, run_b)]
    trained_w = [json.loads((d / "metrics.json").read_text())["mix_weight"] for d in dirs]
    weight_a = trained_w[0] if weight_a is None else weight_a
    weight_b = trained_w[1] if weight_b is None else weight_b
    (weights_a, a), (weights_b, b) = (cross_validation.run_ranks(d, epoch) for d in dirs)
    result = cross_validation.compare_sets(a, b, weights_a, weights_b, weight_a, weight_b)
    _print_comparison(f"{run_b} (mix {weight_b:g}) − {run_a} (mix {weight_a:g}), paired over meshes, epoch: {epoch}:",
                      result, sorted(set(a) - set(b)), sorted(set(b) - set(a)))
    return result

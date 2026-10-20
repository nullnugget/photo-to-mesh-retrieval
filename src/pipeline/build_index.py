"""The index step: embed the gallery of a trained checkpoint once, save it, and check it.

Steps for one checkpoint (the best or the last checkpoint of a training run):

1. load the checkpoint and its training record (`runs/<run_name>/metrics.json`: split, datasets);
2. for each dataset: embed every view and every mesh of the renders feature file with the heads,
   and save the index next to the checkpoint (`src/retrieval/index.py`);
3. check: for a dataset of the training, search the val photos of the training split with the
   index, and compare the R@1 with the validation R@1 that the training logged for the epoch of the
   checkpoint (`train_log.csv`). They must be equal: the index and the validation use the same
   heads, the same features, and the same scores.

Outputs (a mini run reads the checkpoints of a mini training, `test_outputs/train_mini/`):
    weights/checkpoints/<run_name>/embedding_heads_<best|last>_gallery_<dataset>.npz
"""

import csv
import datetime
import json
from pathlib import Path

import numpy as np

from src import paths, runs
from src.data.splits import load_split
from src.models.feature_store import read_features
from src.pipeline import extract_features, train_heads
from src.retrieval import metrics
from src.retrieval.evaluate import correct_columns
from src.retrieval.index import build_index, index_file, load_heads, mix_scores, save_index, search_scores

CHECKPOINT_KINDS = ("best", "last")


def checkpoint_file(run_name: str, which: str = "best", mini: bool = False) -> Path:
    """Return the checkpoint file of a training run ("best" or "last")."""
    if which not in CHECKPOINT_KINDS:
        raise ValueError(f"which must be one of {CHECKPOINT_KINDS}, got {which!r}")
    path = train_heads.output_dirs(mini)["checkpoints"] / run_name / f"embedding_heads_{which}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint {path}")
    return path


def training_record(run_name: str, mini: bool = False) -> dict:
    """Return the metrics.json of a training run (it names the split and the datasets of the training)."""
    return json.loads((train_heads.output_dirs(mini)["runs"] / run_name / "metrics.json").read_text())


def logged_val_r1(run_name: str, epoch: int, mini: bool = False) -> dict[str, float]:
    """Return the validation R@1 of each val query set that the training logged for one epoch."""
    with open(train_heads.output_dirs(mini)["runs"] / run_name / "train_log.csv", newline="") as f:
        row = next(r for r in csv.DictReader(f) if int(r["epoch"]) == epoch)
    return {key[len("val_"):-len("_R@1")]: float(value) for key, value in row.items()
            if key.startswith("val_") and key.endswith("_R@1") and not key.endswith("trained_R@1")}


def run(run_name: str, datasets: list[str] | None = None, which: str = "best", mini: bool = False,
        device: str = "cuda") -> dict:
    """Build and check the index of one checkpoint for each dataset; return the check results.

    Args:
        run_name: The training run, for example "2026-10-20_pix3d_dinov3_vitb16_seen_meshes_mix05".
        datasets: Datasets of the gallery (None = the datasets of the training).
        which: "best" or "last" checkpoint.
        mini: True = a checkpoint of a mini training (test_outputs/train_mini/).
        device: "cuda" or "cpu".
    """
    checkpoint_path = checkpoint_file(run_name, which, mini)
    checkpoint = runs.load_checkpoint(checkpoint_path)
    meta, record = checkpoint["meta"], training_record(run_name, mini)
    cfg = meta["config"]
    backbone, recipe = meta["backbone"], cfg["backbone"]["feature_recipe"]
    # A run from before the score mix validated the trained score alone: mix weight 1.0.
    scoring = {"alpha": cfg["train"]["scoring"]["alpha"], "mix_weight": cfg["train"]["scoring"].get("mix_weight", 1.0),
               "zero_shot": cfg["backbone"]["zero_shot"]}
    features_root = extract_features.output_root(False)  # the real feature files (as in the training)
    logged = logged_val_r1(run_name, meta["epoch"], mini)
    commit, _ = runs.git_state()
    heads = load_heads(checkpoint, device)
    checks = {}
    for dataset in datasets or record["datasets"]:
        renders_file = paths.features_file(backbone, dataset, "renders", features_root)
        renders = read_features(renders_file)
        info = {
            "dataset": dataset,
            "checkpoint": str(checkpoint_path.relative_to(paths.ROOT)),
            "checkpoint_sha256": runs.file_sha256(checkpoint_path),
            "training_run": run_name,
            "training_split": record["split"],
            "epoch": meta["epoch"],
            "renders_file": str(renders_file.relative_to(paths.ROOT)),
            "renders_sha256": runs.file_sha256(renders_file),
            "backbone": backbone,
            "recipe": recipe,
            **scoring,
            "built": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": commit,
        }
        index = build_index(heads, renders, recipe, info, device)
        path = save_index(index, index_file(checkpoint_path, dataset))
        size_mb = path.stat().st_size / 1e6
        print(f"{dataset}: {len(index.mesh_ids)} meshes, {index.views.shape[0] * index.views.shape[1]:,} view embeddings "
              f"+ {len(index.meshes)} mesh embeddings ({index.views.shape[-1]} numbers each), {size_mb:.1f} MB "
              f"-> {path.relative_to(paths.ROOT)}", flush=True)

        if dataset not in record["datasets"]:
            continue  # no validation of this dataset in the training: nothing to compare with
        wanted = load_split(paths.split_file(dataset, record["split"]))["parts"]["val"]
        for query_set in extract_features.QUERY_SETS[dataset]:
            table = read_features(paths.features_file(backbone, dataset, query_set, features_root))
            present = set(table.ids.tolist())
            table = table.select([q for q in wanted if q in present])
            correct = correct_columns(table.labels["mesh_id"], index.mesh_ids)
            final = mix_scores(search_scores(index, heads, table.vectors(recipe), scoring["alpha"], scoring["zero_shot"], device),
                               scoring["mix_weight"])
            r1 = metrics.recall_at_k(metrics.correct_ranks(final, correct), 1)
            name = f"{dataset}_{query_set}"
            expected = logged.get(name)
            equal = expected is not None and round(r1, 4) == expected
            checks[name] = {"index_R@1": r1, "logged_val_R@1": expected, "equal": equal}
            print(f"  check {name}: index val R@1 {100 * r1:.2f}, training log {100 * expected:.2f} "
                  f"(epoch {meta['epoch']}) -> {'equal' if equal else 'DIFFERENT'}", flush=True)
    return {"checkpoint": str(checkpoint_path), "checks": checks}

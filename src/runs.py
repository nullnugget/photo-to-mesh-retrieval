"""Run bookkeeping: seeds, run names, run folders, the results registry, and checkpoints.

A "run" is one training or evaluation execution with one configuration. Each run
has one name, and that name is used in two places:

- `runs/<run_name>/`                Text records: configuration, log, metrics.
                                    These are small and tracked in git.
- `weights/checkpoints/<run_name>/` Binary files: the trained heads and the
                                    gallery index. These are large and not in git.

Five rules keep runs traceable:

1. One name for both places. The same folder name in `runs/` and in
   `weights/checkpoints/` shows which records belong to which checkpoint.
2. One checkpoint file holds all three embedding heads (query projector, view
   projector, view aggregator). The heads only work as a set, because they are
   trained together into one shared embedding space. One file prevents a mix of
   heads from different runs.
3. Each checkpoint describes itself. It stores the run name, the configuration,
   the git commit, the backbone, the feature file and its SHA-256 fingerprint,
   the embedding size, the epoch, and the validation recall@1. A copied
   checkpoint thus still tells where it comes from.
4. Loading checks compatibility. A checkpoint is useless with a different
   backbone, embedding size, or feature file, so `load_checkpoint` stops with an
   error on any such difference.
5. Each run adds one row to `runs/registry.csv`. The registry is the results
   table of the project.
"""

import csv
import datetime
import hashlib
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import paths
from src.config import save_config

# Columns of runs/registry.csv, in order. A row may leave any column empty.
# One row is one result: one model (zero-shot backbone features, or trained heads), evaluated on one
# group of photos with one query set, one feature recipe, and one scoring. An evaluation of several
# settings writes several rows with the same run name.
REGISTRY_COLUMNS = [
    "run_name",
    "date",
    "git_commit",
    "git_dirty",
    "mode",  # "zero_shot" (raw backbone features) or "trained" (embedding heads)
    "datasets",
    "split",  # the split of the photos and of the training; empty for the test-free photos
    "photos",  # the photo group: "test_free", "train", "val", or "test"
    "photo_count",
    "query_set",  # "queries_reference" (the dataset's own masks) or "queries_sam" (SAM masks)
    "backbone",
    "recipe",  # feature recipe: "cls", "masked_mean", or "cls_masked_mean"
    "scoring",  # "view", "mesh", or "combined"
    "alpha",  # weight of the view score in the combined score
    "mix_weight",  # trained runs: final = mix_weight × trained + (1 − mix_weight) × zero-shot score
    "render_config",
    "lighting",
    "temperature",
    "category_weight",
    "epochs",
    "best_epoch",
    "val_R@1",  # trained runs: validation R@1 of the selected checkpoint
    "R@1",
    "R@5",
    "R@10",
    "cat@1",  # share of photos whose first result has the right category
    "macro_R@1",  # mean of the per-category R@1 values
    "checkpoint_path",
    "notes",
]

# Columns of earlier registry versions and their new names. A registry with an older header is
# rewritten with the current header before a row is added (see `append_registry`).
RENAMED_REGISTRY_COLUMNS = {"test_R@1": "R@1", "test_R@5": "R@5", "test_R@10": "R@10", "test_cat_acc@1": "cat@1"}

# Names of the three embedding heads inside a checkpoint file.
HEAD_KEYS = ("query_projector", "view_projector", "view_aggregator")

# Metadata that must be equal between a checkpoint and the current setup before the checkpoint is used.
COMPATIBILITY_KEYS = ("backbone", "embedding_dim", "feature_file_sha256")


class CheckpointMismatch(RuntimeError):
    """The checkpoint was made with a different backbone, embedding size, or feature file."""


# --------------------------------------------------------------------------------------------
# Seeds and code version
# --------------------------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every random number generator that the project uses.

    Args:
        seed: The seed value.
        deterministic: If True, also make cuDNN choose deterministic algorithms.
            This makes GPU results repeatable, but can make them slower.
    """
    random.seed(seed)  # Python's own random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch on the CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch on all GPUs (no effect without a GPU)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_state(repo: Path = paths.ROOT) -> tuple[str, bool]:
    """Return the current git commit and whether the working tree has uncommitted changes.

    Returns:
        (commit hash, has uncommitted changes). Without git, ("unknown", False).
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        # "git status --porcelain" prints one line for each changed file, and nothing if the tree is clean.
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


# --------------------------------------------------------------------------------------------
# Run names and run folders
# --------------------------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Make a text safe for a folder name: lowercase, letters and digits, single underscores.

    Example: "L2 test" -> "l2_test".
    """
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def make_run_name(
    dataset: str,
    backbone: str,
    tag: str,
    date: datetime.date | None = None,
    runs_dir: Path = paths.RUNS_DIR,
    checkpoints_dir: Path = paths.CHECKPOINTS_DIR,
) -> str:
    """Make a new, unused run name of the form `<YYYY-MM-DD>_<dataset>_<backbone>_<tag>`.

    If the name is already used in `runs/` or in `weights/checkpoints/`, the
    function adds `_2`, `_3`, and so on, so that no earlier run is overwritten.

    Example: "2026-10-22_pix3d_dinov3_vitb16_l2".
    """
    date = date or datetime.date.today()
    base = f"{date.isoformat()}_{_slug(dataset)}_{_slug(backbone)}_{_slug(tag)}"
    name, number = base, 1
    while (runs_dir / name).exists() or (checkpoints_dir / name).exists():
        number += 1
        name = f"{base}_{number}"
    return name


def create_run(
    cfg: dict,
    dataset: str,
    backbone: str,
    tag: str,
    runs_dir: Path = paths.RUNS_DIR,
    checkpoints_dir: Path = paths.CHECKPOINTS_DIR,
) -> dict[str, Any]:
    """Start a new run: choose its name, make its two folders, and save its configuration.

    Returns:
        {"run_name": ..., "run_dir": runs/<name>, "checkpoint_dir": weights/checkpoints/<name>}
    """
    name = make_run_name(dataset, backbone, tag, runs_dir=runs_dir, checkpoints_dir=checkpoints_dir)
    run_dir = paths.ensure_dir(runs_dir / name)
    checkpoint_dir = paths.ensure_dir(checkpoints_dir / name)
    # The saved copy records the exact settings of this run, including command-line overrides.
    save_config(cfg, run_dir / "config.yaml")
    return {"run_name": name, "run_dir": run_dir, "checkpoint_dir": checkpoint_dir}


# --------------------------------------------------------------------------------------------
# Results registry
# --------------------------------------------------------------------------------------------


def _upgrade_registry(registry: Path) -> None:
    """Rewrite a registry whose header is not REGISTRY_COLUMNS, keeping every value.

    Renamed columns (RENAMED_REGISTRY_COLUMNS) move to their new names. A row of the first version
    held only test results, so it gets photos = "test" and mode = "trained". A column that no longer
    exists is an error, so that no value is lost silently.
    """
    with open(registry, newline="") as f:
        rows = list(csv.DictReader(f))
    upgraded = []
    for row in rows:
        old_test_row = "test_R@1" in row
        row = {RENAMED_REGISTRY_COLUMNS.get(column, column): value for column, value in row.items()}
        unknown = set(row) - set(REGISTRY_COLUMNS)
        if unknown:
            raise KeyError(f"{registry}: columns {sorted(unknown)} are not in REGISTRY_COLUMNS")
        if old_test_row and row.get("R@1"):
            row.setdefault("photos", "test")
            row.setdefault("mode", "trained")
        upgraded.append({column: row.get(column, "") for column in REGISTRY_COLUMNS})
    with open(registry, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(upgraded)


def append_registry(row: dict[str, Any], registry: Path = paths.REGISTRY_CSV) -> Path:
    """Add one row to the results registry (a CSV file).

    The header is written only when the file is new. A file with an older header is first rewritten
    with the current header (`_upgrade_registry`). Columns that the row does not give stay empty.
    A key that is not in REGISTRY_COLUMNS is an error, so that a typing error cannot create a
    silent new column.
    """
    unknown = set(row) - set(REGISTRY_COLUMNS)
    if unknown:
        raise KeyError(f"unknown registry columns: {sorted(unknown)}")
    registry.parent.mkdir(parents=True, exist_ok=True)
    is_new = not registry.exists() or registry.stat().st_size == 0
    if not is_new:
        with open(registry, newline="") as f:
            header = next(csv.reader(f))
        if header != REGISTRY_COLUMNS:
            _upgrade_registry(registry)
    with open(registry, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in REGISTRY_COLUMNS})
    return registry


def read_registry(registry: Path = paths.REGISTRY_CSV) -> list[dict[str, str]]:
    """Return all rows of the registry as dicts (an empty list if the file does not exist)."""
    if not registry.exists():
        return []
    with open(registry, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 fingerprint of a file. The file is read in 1 MiB chunks to keep memory low."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(
    run_name: str,
    cfg: dict,
    backbone: str,
    feature_file: Path | list[Path],
    embedding_dim: int,
    epoch: int,
    val_r1: float,
) -> dict[str, Any]:
    """Build the self-description that is saved inside each checkpoint.

    Args:
        run_name: Name of the run that made the checkpoint.
        cfg: Full configuration of the run.
        backbone: Name of the frozen backbone, for example "dinov3_vitb16".
        feature_file: Feature store file(s) that the heads were trained on (one renders file for
            each dataset of the training).
        embedding_dim: Length of the embeddings that the heads produce.
        epoch: Epoch of this checkpoint.
        val_r1: Validation recall@1 at this epoch.

    With several files, "feature_file_sha256" is the SHA-256 of their SHA-256 values in name order,
    so that one value still identifies the whole set; "feature_files" lists each file.
    """
    commit, dirty = git_state()
    files = [Path(f) for f in (feature_file if isinstance(feature_file, list) else [feature_file])]
    # Store the feature file paths relative to the repository root when possible, so they are valid on other machines.
    shown = {str(f.relative_to(paths.ROOT) if f.is_relative_to(paths.ROOT) else f): file_sha256(f) for f in files}
    combined = next(iter(shown.values())) if len(shown) == 1 else hashlib.sha256("".join(shown[k] for k in sorted(shown)).encode()).hexdigest()
    return {
        "run_name": run_name,
        "config": cfg,
        "git_commit": commit,
        "git_dirty": dirty,
        "backbone": backbone,
        "feature_file": ", ".join(shown),
        "feature_file_sha256": combined,
        "feature_files": shown,
        "embedding_dim": int(embedding_dim),
        "epoch": int(epoch),
        "val_r1": float(val_r1),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def save_checkpoint(path: Path, heads: dict[str, dict], meta: dict[str, Any]) -> Path:
    """Save the three embedding heads and their metadata in one file.

    Args:
        path: Output file, normally `weights/checkpoints/<run_name>/embedding_heads_best.pt`.
        heads: The state dicts of the three heads, under the names in HEAD_KEYS.
        meta: The self-description from `checkpoint_metadata`.

    The file contains the keys "query_projector", "view_projector", "view_aggregator", and "meta".
    """
    missing = [key for key in HEAD_KEYS if key not in heads]
    if missing:
        raise KeyError(f"a checkpoint needs all three heads; missing: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**{key: heads[key] for key in HEAD_KEYS}, "meta": meta}, path)
    return path


def load_checkpoint(path: Path, expected: dict[str, Any] | None = None, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint and, optionally, check that it matches the current setup.

    Args:
        path: The checkpoint file.
        expected: Values that the checkpoint must have. Any of: "backbone",
            "embedding_dim", "feature_file_sha256". None skips the check.
        map_location: Device for the loaded tensors.

    Raises:
        CheckpointMismatch: If any expected value differs from the checkpoint metadata.
    """
    # weights_only=False because the metadata contains plain Python objects (dicts, strings), not only tensors.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    meta = checkpoint.get("meta", {})
    for key in COMPATIBILITY_KEYS:
        if expected and key in expected and meta.get(key) != expected[key]:
            raise CheckpointMismatch(
                f"{path}: {key} is {meta.get(key)!r} in the checkpoint, but {expected[key]!r} is expected"
            )
    return checkpoint


if __name__ == "__main__":
    # Stand-alone check (run from the repository root: python -m src.runs).
    # It prints the git state, an example run name, and the rows of the registry. It writes nothing.
    commit, dirty = git_state()
    print(f"git: {commit[:12]} (uncommitted changes: {dirty})")
    print("example run name:", make_run_name("pix3d", "dinov3_vitb16", "check"))
    rows = read_registry()
    print(f"registry: {paths.REGISTRY_CSV.relative_to(paths.ROOT)}, {len(rows)} row(s)")
    for row in rows:
        print("  ", row["run_name"], "|", row["photos"], "R@1:", row["R@1"])

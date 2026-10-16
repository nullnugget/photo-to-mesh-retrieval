"""Run bookkeeping (design guide 02, section 16.5).

A run has one name, used in two places:

- `runs/<run_name>/`: configuration, log, metrics (text, tracked in git);
- `weights/checkpoints/<run_name>/`: checkpoint and gallery index (binary, not in git).

The rules:

1. One run name for both locations (`make_run_name`, `create_run`).
2. One checkpoint file holds all three embedding heads (`save_checkpoint`).
3. Each checkpoint describes itself: run name, configuration, git commit, backbone,
   feature file and its SHA-256, embedding size, epoch, validation R@1.
4. Loading checks compatibility and stops on a mismatch (`load_checkpoint`).
5. Each run adds one row to `runs/registry.csv` (`append_registry`).
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

REGISTRY_COLUMNS = [
    "run_name",
    "date",
    "git_commit",
    "git_dirty",
    "split",
    "datasets",
    "backbone",
    "render_config",
    "lighting",
    "temperature",
    "category_weight",
    "alpha",
    "epochs",
    "best_epoch",
    "val_R@1",
    "test_R@1",
    "test_R@5",
    "test_R@10",
    "test_cat_acc@1",
    "checkpoint_path",
    "notes",
]

HEAD_KEYS = ("query_projector", "view_projector", "view_aggregator")

# Metadata keys that must match between a checkpoint and the current configuration
COMPATIBILITY_KEYS = ("backbone", "embedding_dim", "feature_file_sha256")


class CheckpointMismatch(RuntimeError):
    """The checkpoint does not match the current backbone, embedding size, or feature file."""


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA). `deterministic=True` also makes cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_state(repo: Path = paths.ROOT) -> tuple[str, bool]:
    """Return (commit hash, True if the working tree has uncommitted changes). ("unknown", False) without git."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def _slug(text: str) -> str:
    """Lowercase, with only letters, digits, and single underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def make_run_name(
    dataset: str,
    backbone: str,
    tag: str,
    date: datetime.date | None = None,
    runs_dir: Path = paths.RUNS_DIR,
    checkpoints_dir: Path = paths.CHECKPOINTS_DIR,
) -> str:
    """Return `<YYYY-MM-DD>_<dataset>_<backbone>_<tag>`.

    If that name exists in `runs/` or in `weights/checkpoints/`, add `_2`, `_3`, and so on.
    """
    date = date or datetime.date.today()
    base = f"{date.isoformat()}_{_slug(dataset)}_{_slug(backbone)}_{_slug(tag)}"
    name, n = base, 1
    while (runs_dir / name).exists() or (checkpoints_dir / name).exists():
        n += 1
        name = f"{base}_{n}"
    return name


def create_run(
    cfg: dict,
    dataset: str,
    backbone: str,
    tag: str,
    runs_dir: Path = paths.RUNS_DIR,
    checkpoints_dir: Path = paths.CHECKPOINTS_DIR,
) -> dict[str, Any]:
    """Make the two run folders, save the configuration, and return the run name and folders."""
    name = make_run_name(dataset, backbone, tag, runs_dir=runs_dir, checkpoints_dir=checkpoints_dir)
    run_dir = paths.ensure_dir(runs_dir / name)
    ckpt_dir = paths.ensure_dir(checkpoints_dir / name)
    save_config(cfg, run_dir / "config.yaml")
    return {"run_name": name, "run_dir": run_dir, "checkpoint_dir": ckpt_dir}


def append_registry(row: dict[str, Any], registry: Path = paths.REGISTRY_CSV) -> Path:
    """Add one row to the registry. Write the header if the file is new. Unknown columns are an error."""
    unknown = set(row) - set(REGISTRY_COLUMNS)
    if unknown:
        raise KeyError(f"unknown registry columns: {sorted(unknown)}")
    registry.parent.mkdir(parents=True, exist_ok=True)
    is_new = not registry.exists() or registry.stat().st_size == 0
    with open(registry, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in REGISTRY_COLUMNS})
    return registry


def read_registry(registry: Path = paths.REGISTRY_CSV) -> list[dict[str, str]]:
    """Return all registry rows (an empty list if the file does not exist)."""
    if not registry.exists():
        return []
    with open(registry, newline="") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_metadata(
    run_name: str,
    cfg: dict,
    backbone: str,
    feature_file: Path,
    embedding_dim: int,
    epoch: int,
    val_r1: float,
) -> dict[str, Any]:
    """Build the self-description that goes into each checkpoint (rule 3)."""
    commit, dirty = git_state()
    return {
        "run_name": run_name,
        "config": cfg,
        "git_commit": commit,
        "git_dirty": dirty,
        "backbone": backbone,
        "feature_file": str(Path(feature_file).relative_to(paths.ROOT)) if Path(feature_file).is_relative_to(paths.ROOT) else str(feature_file),
        "feature_file_sha256": file_sha256(Path(feature_file)),
        "embedding_dim": int(embedding_dim),
        "epoch": int(epoch),
        "val_r1": float(val_r1),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def save_checkpoint(path: Path, heads: dict[str, dict], meta: dict[str, Any]) -> Path:
    """Save the three heads (state dicts) and the metadata in one file (rule 2)."""
    missing = [k for k in HEAD_KEYS if k not in heads]
    if missing:
        raise KeyError(f"checkpoint needs all three heads; missing: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**{k: heads[k] for k in HEAD_KEYS}, "meta": meta}, path)
    return path


def load_checkpoint(path: Path, expected: dict[str, Any] | None = None, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint. If `expected` is given, stop with `CheckpointMismatch` on any difference (rule 4).

    `expected` can contain any of: backbone, embedding_dim, feature_file_sha256.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt.get("meta", {})
    for key in COMPATIBILITY_KEYS:
        if expected and key in expected and meta.get(key) != expected[key]:
            raise CheckpointMismatch(
                f"{path}: {key} is {meta.get(key)!r} in the checkpoint, but {expected[key]!r} is expected"
            )
    return ckpt


if __name__ == "__main__":
    # Stand-alone check: show the git state, a run name, and the current registry rows.
    commit, dirty = git_state()
    print(f"git: {commit[:12]} (uncommitted changes: {dirty})")
    print("example run name:", make_run_name("pix3d", "dinov3_vitb16", "check"))
    rows = read_registry()
    print(f"registry: {paths.REGISTRY_CSV.relative_to(paths.ROOT)}, {len(rows)} row(s)")
    for r in rows:
        print("  ", r["run_name"], "| test R@1:", r["test_R@1"])

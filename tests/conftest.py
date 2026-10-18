"""Shared pytest settings for the v2 tests.

- Tests marked `gpu` are skipped when no CUDA GPU is available.
- Tests marked `data` are skipped when the Pix3D data is not in `data/raw/pix3d/`.
- Tests marked `weights` are skipped when the segmentation weights are not in `weights/`.
- The automatic tests never write into the project folders; they use pytest's `tmp_path`.
"""

import pytest
import torch

from src import paths
from src.config import load_config


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    skip_data = pytest.mark.skip(reason=f"dataset missing: {paths.PIX3D_JSON}")
    segment_cfg = load_config()["segment"]
    weight_files = [paths.resolve(segment_cfg[m]["folder"]) / "model.safetensors" for m in ("detector", "segmenter")]
    skip_weights = pytest.mark.skip(reason="segmentation weights missing (scripts/00_download_weights.py)")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        if "data" in item.keywords and not paths.PIX3D_JSON.exists():
            item.add_marker(skip_data)
        if "weights" in item.keywords and not all(f.exists() for f in weight_files):
            item.add_marker(skip_weights)


@pytest.fixture
def scratch_run_dirs(tmp_path):
    """Temporary `runs/` and `weights/checkpoints/` folders for run-tool tests."""
    return {"runs_dir": tmp_path / "runs", "checkpoints_dir": tmp_path / "checkpoints"}

"""Shared pytest settings for the v2 tests.

- Tests marked `gpu` are skipped when no CUDA GPU is available.
- Tests marked `data` are skipped when the Pix3D data is not in `data/raw/pix3d/`.
- The automatic tests never write into the project folders; they use pytest's `tmp_path`.
"""

import pytest
import torch

from src import paths


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    skip_data = pytest.mark.skip(reason=f"dataset missing: {paths.PIX3D_JSON}")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        if "data" in item.keywords and not paths.PIX3D_JSON.exists():
            item.add_marker(skip_data)


@pytest.fixture
def scratch_run_dirs(tmp_path):
    """Temporary `runs/` and `weights/checkpoints/` folders for run-tool tests."""
    return {"runs_dir": tmp_path / "runs", "checkpoints_dir": tmp_path / "checkpoints"}

"""The weights download step: the pretrained detector and segmenter, from the Hugging Face Hub.

For each chosen model in `configs/segment.yaml` (by default `segment.detector` and `segment.segmenter`;
`segment.segmenter_large` only on request), this step
downloads the listed files from the model repository at the fixed commit ("revision") into the
model's folder under `weights/`, and checks the size of every file. Later code loads the models
only from these folders, so nothing is downloaded while the pipeline runs, and the exact weights
are known.

Files that exist with the right size are skipped. Each file is written through a temporary name,
so an interrupted download leaves no file that looks complete. A small record,
`download_record.json`, in each folder names the repository, the commit, and the files.

The DINOv3 and DINOv2 backbone weights are not downloaded here: DINOv3 needs a request to Meta,
and both are placed in `weights/dinov3/` and `weights/dinov2/` by hand (see the README).
"""

import json
from pathlib import Path

from src import paths
from src.data.http import Fetcher

HUB_URL = "https://huggingface.co"
MODELS = ("detector", "segmenter", "segmenter_large")  # every model that can be downloaded
DEFAULT_MODELS = ("detector", "segmenter")  # the models the pipeline uses now


def file_url(hub_repo: str, revision: str, name: str) -> str:
    """Return the download URL of one file of a model repository at a fixed commit."""
    return f"{HUB_URL}/{hub_repo}/resolve/{revision}/{name}"


def download_model(fetcher: Fetcher, model_cfg: dict) -> dict:
    """Download the files of one model. Returns {"folder", "downloaded", "present", "bytes"}.

    Raises:
        ValueError: If a downloaded file does not have the size given in the configuration.
    """
    folder = paths.resolve(model_cfg["folder"])
    downloaded, present, total = [], [], 0
    for name, size in model_cfg["files"].items():
        target = folder / name
        if target.exists() and target.stat().st_size == size:
            present.append(name)
            continue
        total += fetcher.save(file_url(model_cfg["hub_repo"], model_cfg["revision"], name), target)
        if target.stat().st_size != size:
            target.unlink()
            raise ValueError(f"{model_cfg['hub_repo']}/{name}: expected {size} bytes, got another size")
        downloaded.append(name)
    record = {"hub_repo": model_cfg["hub_repo"], "revision": model_cfg["revision"], "files": model_cfg["files"]}
    (folder / "download_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return {"folder": str(folder.relative_to(paths.ROOT)), "downloaded": downloaded, "present": present, "bytes": total}


def run(cfg: dict, models: tuple[str, ...] = DEFAULT_MODELS) -> dict:
    """Download the weights of the chosen models (`cfg["segment"]`), and return a report."""
    fetcher = Fetcher(retries=5, timeout=120.0)
    report = {}
    for model in models:
        result = download_model(fetcher, cfg["segment"][model])
        report[model] = result
        print(f"{model}: {len(result['downloaded'])} downloaded ({result['bytes'] / 1e6:.0f} MB), "
              f"{len(result['present'])} already there -> {result['folder']}")
    return report

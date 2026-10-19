"""The training photos of a split part, as a PyTorch dataset of randomly changed query images.

Each training photo has one or more prepared query images (224 × 224, grey background, tight crop),
named by the mask that cut them out (`train.query_images` in `configs/train.yaml`):

- "reference": the catalog query image. For Pix3D it is cut out with the dataset's own mask.
- "sam": the query image cut out with the SAM mask. For Pix3D the feature step makes it
  (`data/derived/masks/pix3d_sam_queries/`); for ABO it is the catalog image (ABO has no other mask).

At each use of a photo, the dataset picks one of its images at random (for Pix3D: the reference or
the SAM image, 50/50), changes it at random (`src/data/augment.py`), and returns it with the number
of its mesh and of its category. So the heads see each photo with two outlines and new changes in
every epoch. The real SAM errors teach the heads the masks of real use; the reference images keep
the clean outlines of the main evaluation.

Why prepared images and not the original photos: the originals are large (Pix3D up to several
thousand pixels), and cutting them out at every use is about 10 to 20 times too slow for training.

Random numbers: every data loader worker gets its own NumPy generator, seeded from the worker seed
that PyTorch gives it (a new one in every epoch). So a run with the same seed repeats exactly, and
two workers never repeat each other's changes.
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from src import paths
from src.data.augment import QueryAugment
from src.data.catalog import read_table
from src.data.splits import load_split
from src.pipeline.extract_features import sam_query_file

IMAGE_KINDS = ("reference", "sam")


@dataclass(frozen=True)
class TrainingPhoto:
    """One training photo and the prepared query images that it can use."""

    query_id: str
    dataset: str
    category: str
    mesh_id: str
    images: tuple[str, ...]  # absolute paths; one is picked at random at each use


def query_image_file(row: dict, kind: str) -> Path:
    """Return the prepared query image of one catalog row and one kind ("reference" or "sam")."""
    if kind not in IMAGE_KINDS:
        raise ValueError(f"kind must be one of {IMAGE_KINDS}, got {kind!r}")
    if kind == "sam" and row["dataset"] == "pix3d":
        return sam_query_file(paths.MASKS_DIR, row)  # made by the feature step from the SAM mask
    return paths.resolve(row["query_path"])  # Pix3D: the dataset's mask; ABO: the SAM mask


def training_photos(datasets: list[str], split: str, part: str, query_images_cfg: dict) -> tuple[list[TrainingPhoto], dict]:
    """Return the photos of one split part and a short report.

    Args:
        datasets: For example ["pix3d"] or ["pix3d", "abo"].
        split: The split name, for example "seen_meshes".
        part: "train" (normally), "val", or "test".
        query_images_cfg: `train.query_images`: dataset -> the image kinds to use.

    Returns:
        (photos, report). A photo keeps the images that exist on disk; a photo without any image is
        left out. The report counts the photos, the images of each kind, and the left-out photos.
    """
    rows = {row["query_id"]: row for row in read_table(paths.QUERIES_CSV)}
    photos, report = [], {"photos": 0, "images": Counter(), "missing_images": Counter(), "left_out": []}
    for dataset in datasets:
        kinds = query_images_cfg[dataset]
        for query_id in load_split(paths.split_file(dataset, split))["parts"][part]:
            row = rows[query_id]
            images = []
            for kind in kinds:
                file = query_image_file(row, kind)
                if file.exists():
                    images.append(str(file))
                    report["images"][f"{dataset}_{kind}"] += 1
                else:
                    report["missing_images"][f"{dataset}_{kind}"] += 1
            if images:
                photos.append(TrainingPhoto(query_id, dataset, row["category"], row["mesh_id"], tuple(images)))
            else:
                report["left_out"].append(query_id)
    report["photos"] = len(photos)
    report["images"], report["missing_images"] = dict(report["images"]), dict(report["missing_images"])
    return photos, report


def loader_worker_init(_worker_id: int) -> None:
    """Prepare a data loader worker process: all CPU cores, and one PyTorch thread.

    1. All CPU cores. A worker is a copy (fork) of the main process, and it inherits the core
       binding of the main thread. The OpenMP runtime of the CPU math libraries can bind the main
       thread to one core (2 logical CPUs) when it starts; then all workers share these 2 CPUs.
       Measured: 111 images per second with 4 workers on 2 CPUs, 273 on all 16. Whether the binding
       exists depends on what the main process did before (the order of the imports, a notebook),
       so the worker always sets all cores again.
    2. One PyTorch thread. The image changes use small PyTorch operations; with the default (one
       thread per core in each worker) the workers slow each other down.
    """
    if hasattr(os, "sched_setaffinity"):  # Linux; other systems do not bind this way
        os.sched_setaffinity(0, range(os.cpu_count()))
    torch.set_num_threads(1)


class TrainingQueries(Dataset):
    """Randomly changed query images of the training photos, with their mesh and category numbers."""

    def __init__(self, photos: list[TrainingPhoto], augment: QueryAugment, mesh_index: dict[str, int],
                 category_index: dict[str, int], seed: int):
        """
        Args:
            photos: The training photos (`training_photos`).
            augment: The random changes.
            mesh_index: mesh_id -> column of the mesh in the training gallery.
            category_index: category -> category number.
            seed: Seed of the random changes when the data loader has no worker processes.
        """
        self.photos = photos
        self.augment = augment
        self.mesh_index = mesh_index
        self.category_index = category_index
        self.seed = seed
        self._rng_key, self._rng = None, None

    def __len__(self) -> int:
        return len(self.photos)

    def _generator(self) -> np.random.Generator:
        """The NumPy generator of the current process: one per worker, new in every epoch."""
        info = get_worker_info()
        key = info.seed if info is not None else self.seed  # PyTorch gives each worker a new seed in each epoch
        if key != self._rng_key:
            self._rng_key, self._rng = key, np.random.default_rng(key % 2**63)
        return self._rng

    def __getitem__(self, index: int) -> dict:
        rng = self._generator()
        photo = self.photos[index]
        file = photo.images[int(rng.integers(len(photo.images)))]  # for Pix3D: the reference or the SAM image
        image = self.augment(np.array(Image.open(file).convert("RGB")), rng)  # np.array: a writable copy
        return {
            "image": torch.from_numpy(image),  # uint8 (size, size, 3)
            "mesh": self.mesh_index[photo.mesh_id],
            "category": self.category_index[photo.category],
            "index": index,
        }

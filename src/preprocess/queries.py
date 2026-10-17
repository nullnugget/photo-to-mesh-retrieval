"""Query images: read a photo and its object mask, and make the query image.

A query image is a photo after preparation. `crop_to_square` replaces the background with grey,
crops tightly around the object, pads to a square, and resizes to 224 x 224.

This module does not depend on a dataset. Each job names a photo file and a mask file:

- Pix3D: the mask files that come with the dataset (made by hand by the Pix3D authors).
- ABO: the mask files that the segmentation step makes (Grounding DINO box, then SAM 2.1 mask).

`make_query_images` runs many jobs in parallel worker processes. The worker function is in this
module (and not in a script or a notebook), because worker processes must be able to import it.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src import paths
from src.preprocess.crop import EmptyMaskError, crop_to_square

# Status values in the result of each job.
DONE, SKIPPED, FAILED = "done", "skipped", "failed"


class MaskMismatchError(ValueError):
    """The mask does not have the size of its photo, so it belongs to another image."""


def read_photo_and_mask(photo_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one photo and its mask, ready for the crop.

    Args:
        photo_path: Photo file, relative to the repository root (or absolute).
        mask_path: Mask file (white = object), relative to the repository root (or absolute).

    Returns:
        image: uint8 array of shape (H, W, 3), RGB.
        mask: bool array of shape (H, W); True = object.

    Photos come in several formats and modes (JPEG, PNG, TIFF; grayscale, RGBA, palette).
    Pillow reads all of them, and `convert("RGB")` gives 3 channels in every case.

    Raises:
        MaskMismatchError: If the mask size differs from the photo size. Then the mask file
            belongs to another photo, and resizing cannot repair it.
    """
    with Image.open(paths.resolve(photo_path)) as photo:
        # A palette image with a transparent colour must go through RGBA first; Pillow warns
        # otherwise. The RGB values are the same either way (the mask removes the background).
        image = (photo.convert("RGBA") if photo.mode == "P" else photo).convert("RGB")
    with Image.open(paths.resolve(mask_path)) as mask_file:
        mask = mask_file.convert("L")  # one grey channel, values 0..255
    if mask.size != image.size:
        raise MaskMismatchError(f"{mask_path}: mask size {mask.size} differs from photo size {image.size}")
    # Mask pixels above the middle grey are object pixels.
    return np.asarray(image, dtype=np.uint8), np.asarray(mask) > 127


@dataclass(frozen=True)
class QueryJob:
    """The instructions to make one query image."""

    query_id: str  # for example "pix3d/chair/0001"
    photo_path: str  # relative to the repository root
    mask_path: str  # relative to the repository root
    output_path: str  # absolute path of the PNG file to write
    background_rgb: tuple[int, int, int]  # colour of the background and of the padding
    margin: float  # border around the object, as a fraction of the longer side of its box
    output_size: int  # side length of the query image, in pixels
    overwrite: bool  # True = make the image again if the file exists


def make_query_image(job: QueryJob) -> dict:
    """Make and save one query image.

    Returns:
        {"query_id": ..., "status": "done" | "skipped" | "failed", "message": ...}.
        "skipped" means that the file exists and `job.overwrite` is False. "failed" means that
        the photo cannot be prepared (a mask of another image, or an empty mask); the message
        gives the reason.
    """
    output_path = Path(job.output_path)
    if output_path.exists() and not job.overwrite:
        return {"query_id": job.query_id, "status": SKIPPED, "message": "exists"}
    try:
        image, mask = read_photo_and_mask(job.photo_path, job.mask_path)
        query = crop_to_square(image, mask, tuple(job.background_rgb), job.margin, job.output_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(query).save(output_path)
        return {"query_id": job.query_id, "status": DONE, "message": ""}
    except (MaskMismatchError, EmptyMaskError) as error:
        return {"query_id": job.query_id, "status": FAILED, "message": f"{type(error).__name__}: {error}"}


def _init_worker() -> None:
    """Run once in each worker process: one thread for torch, because the processes already run in parallel."""
    torch.set_num_threads(1)


def default_workers() -> int:
    """Number of worker processes to use when none is given: 8, or fewer on a smaller machine."""
    return min(8, os.cpu_count() or 1)


def make_query_images(jobs: list[QueryJob], workers: int | None = None) -> list[dict]:
    """Make the query images of all jobs, in parallel, and return one result per job (same order).

    Args:
        jobs: The jobs to run.
        workers: Number of worker processes. 1 runs every job in this process (easier to debug).
            None uses `default_workers()`.
    """
    workers = workers or default_workers()
    progress = {"total": len(jobs), "desc": "query images", "unit": "img"}
    if workers <= 1:
        return [make_query_image(job) for job in tqdm(jobs, **progress)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        return list(tqdm(pool.map(make_query_image, jobs, chunksize=16), **progress))

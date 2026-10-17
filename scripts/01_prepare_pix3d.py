"""Prepare the Pix3D query images, the catalog rows, and the data splits.

For every Pix3D photo, this script:

1. reads the photo and its object mask;
2. makes the query image with the shared crop function (grey background, tight square crop,
   224 x 224), and saves it as PNG;
3. writes one row per query and one row per shape into the shared catalog;
4. writes the three split files (image-level, unseen-shape, leave-categories-out).

Photos that cannot be prepared (in Pix3D: two photos whose mask belongs to another image) are
reported and left out of the catalog and the splits.

Outputs (normal run):
    data/derived/queries/pix3d/<category>/<name>.png    query images
    data/derived/catalog/queries.csv, shapes.csv        catalog (Pix3D rows replaced)
    data/derived/catalog/pix3d_report.json              counts, failures, settings
    data/splits/*.json                                  split files

With --mini, only 5 photos per category are prepared, and all outputs go to
test_outputs/prepare_pix3d_mini/ instead, so that the real outputs are not touched.

Usage (from the repository root):
    python scripts/01_prepare_pix3d.py              # all photos
    python scripts/01_prepare_pix3d.py --mini       # quick check
    python scripts/01_prepare_pix3d.py --overwrite  # make the query images again
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.data import catalog, splits  # noqa: E402
from src.data.pix3d import DATASET, MaskMismatchError, load_annotations, photo_records, query_id_of, read_photo_and_mask, shape_records  # noqa: E402
from src.preprocess.crop import EmptyMaskError, crop_to_square  # noqa: E402


def output_dirs(mini: bool) -> dict[str, Path]:
    """Return the output folders: the real ones, or a separate test folder for --mini."""
    if mini:
        base = paths.TEST_OUTPUTS_DIR / "prepare_pix3d_mini"
        return {"queries": base / "queries" / DATASET, "catalog": base / "catalog", "splits": base / "splits"}
    return {"queries": paths.QUERIES_DIR / DATASET, "catalog": paths.CATALOG_DIR, "splits": paths.SPLITS_DIR}


def _init_worker() -> None:
    """Run once in each worker process: one thread for torch, because the processes already run in parallel."""
    torch.set_num_threads(1)


def prepare_one(job: dict) -> dict:
    """Make and save one query image. Runs in a worker process.

    Args:
        job: query_id, photo_path, mask_path, output_path (absolute), background_rgb, margin,
            output_size, overwrite.

    Returns:
        {"query_id": ..., "status": "done" | "skipped" | "failed", "message": ...}
    """
    output_path = Path(job["output_path"])
    if output_path.exists() and not job["overwrite"]:
        return {"query_id": job["query_id"], "status": "skipped", "message": "exists"}
    try:
        image, mask = read_photo_and_mask(job["photo_path"], job["mask_path"])
        query = crop_to_square(image, mask, tuple(job["background_rgb"]), job["margin"], job["output_size"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(query).save(output_path)
        return {"query_id": job["query_id"], "status": "done", "message": ""}
    except (MaskMismatchError, EmptyMaskError) as error:
        return {"query_id": job["query_id"], "status": "failed", "message": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_config_args(parser)
    parser.add_argument("--mini", action="store_true", help="prepare 5 photos per category into test_outputs/")
    parser.add_argument("--overwrite", action="store_true", help="make the query images again if they exist")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="parallel processes")
    args = parser.parse_args()
    cfg = config_from_args(args)
    data_cfg = cfg["data"]
    out = output_dirs(args.mini)
    start = time.perf_counter()

    # 1. Read the annotations and make the photo and shape records.
    annotations = load_annotations()
    photos = photo_records(annotations)
    if args.mini:  # the first 5 photos of each category, in annotation order
        taken: Counter = Counter()
        photos = [p for p in photos if (taken.update([p.category]) or taken[p.category] <= 5)]
    print(f"Pix3D: {len(photos)} photos to prepare{' (mini)' if args.mini else ''}; {args.workers} workers")

    # 2. Make the query images in parallel.
    jobs = [
        {
            "query_id": p.query_id,
            "photo_path": p.photo_path,
            "mask_path": p.mask_path,
            "output_path": str(out["queries"] / p.category / f"{Path(p.photo_path).stem}.png"),
            "background_rgb": data_cfg["background_rgb"],
            "margin": data_cfg["crop_margin"],
            "output_size": data_cfg["image_size"],
            "overwrite": args.overwrite,
        }
        for p in photos
    ]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        results = list(tqdm(pool.map(prepare_one, jobs, chunksize=16), total=len(jobs), desc="queries", unit="img"))
    status = Counter(r["status"] for r in results)
    failed = [r for r in results if r["status"] == "failed"]
    ok_ids = {r["query_id"] for r in results if r["status"] != "failed"}
    print(f"queries: {status['done']} made, {status['skipped']} already there, {status['failed']} failed")
    for r in failed:
        print(f"  left out: {r['query_id']}  ({r['message']})")

    # 3. Catalog rows for the prepared queries, and for the shapes that have at least one query.
    output_of = {job["query_id"]: Path(job["output_path"]) for job in jobs}
    query_rows = [
        {
            "query_id": p.query_id,
            "dataset": DATASET,
            "category": p.category,
            "shape_id": p.shape_id,
            "photo_path": p.photo_path,
            "mask_path": p.mask_path,
            "query_path": str(output_of[p.query_id].relative_to(paths.ROOT)),
            "truncated": p.truncated,
            "occluded": p.occluded,
            "slightly_occluded": p.slightly_occluded,
        }
        for p in photos
        if p.query_id in ok_ids
    ]
    photos_per_shape = Counter(row["shape_id"] for row in query_rows)
    shape_rows = [
        {
            "shape_id": s.shape_id,
            "dataset": DATASET,
            "category": s.category,
            "mesh_path": s.mesh_path,
            "render_dir": str((paths.RENDERS_DIR / s.shape_id).relative_to(paths.ROOT)),  # renders/pix3d/<cat>/<folder>
            "num_photos": photos_per_shape[s.shape_id],
            "num_mesh_variants": s.num_mesh_variants,
        }
        for s in shape_records(annotations)
        if photos_per_shape[s.shape_id] > 0
    ]
    catalog.replace_dataset_rows(out["catalog"] / "queries.csv", query_rows, DATASET, catalog.QUERY_COLUMNS)
    catalog.replace_dataset_rows(out["catalog"] / "shapes.csv", shape_rows, DATASET, catalog.SHAPE_COLUMNS)
    print(f"catalog: {len(query_rows)} query rows, {len(shape_rows)} shape rows -> {out['catalog'].relative_to(paths.ROOT)}")

    # 4. Split files. The image-level split needs the order of ALL photos in pix3d.json (the v1 shuffle).
    split_cfg = data_cfg["splits"]
    seed = split_cfg["seed"]
    made_splits = [
        (
            split_cfg["image_level"]["file"],
            splits.image_level_split(
                [query_id_of(e) for e in annotations],
                query_rows,
                seed=seed,
                v1_holdout_fraction=split_cfg["image_level"]["v1_holdout_fraction"],
                val_fraction_of_train=split_cfg["image_level"]["val_fraction_of_train"],
            ),
        ),
        (
            split_cfg["unseen_shapes"]["file"],
            splits.unseen_shape_split(query_rows, seed=seed, fractions=split_cfg["unseen_shapes"]["fractions"]),
        ),
    ]
    held_out = [c for c in split_cfg["leave_categories_out"]["held_out_pix3d"] if any(r["category"] == c for r in query_rows)]
    if held_out:  # in --mini mode a held-out category can be missing; skip the split then
        made_splits.append(
            (split_cfg["leave_categories_out"]["file"], splits.leave_categories_out_split(query_rows, held_out, seed=seed))
        )
    for file_name, split in made_splits:
        splits.save_split(split, out["splits"] / file_name)
        print(f"split {split['name']}: {split['counts']} -> {(out['splits'] / file_name).relative_to(paths.ROOT)}")

    # 5. Report: counts, failures, settings, and run time.
    report = {
        "dataset": DATASET,
        "mini": args.mini,
        "photos_in_annotations": len(annotations),
        "queries_prepared": len(query_rows),
        "shapes": len(shape_rows),
        "status": dict(status),
        "left_out": failed,
        "queries_per_category": dict(sorted(Counter(r["category"] for r in query_rows).items())),
        "settings": {k: data_cfg[k] for k in ("image_size", "background_rgb", "crop_margin")},
        "splits": {split["name"]: split["counts"] for _, split in made_splits},
        "seconds": round(time.perf_counter() - start, 1),
    }
    report_path = out["catalog"] / "pix3d_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {report_path.relative_to(paths.ROOT)} ({report['seconds']} s)")


if __name__ == "__main__":
    main()

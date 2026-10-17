"""The query step: query images, catalog rows, and split files for one dataset.

For every photo of the dataset, this step:

1. reads the photo and its object mask;
2. makes the query image with the shared crop function (grey background, tight square crop,
   224 x 224), and saves it as PNG;
3. writes one row per query and one row per mesh into the shared catalog;
4. writes the three split files (seen_meshes, unseen_meshes, unseen_categories).

Photos that cannot be prepared (in Pix3D: two photos whose mask belongs to another image) are
reported and left out of the catalog and the splits.

Outputs (normal run):
    data/derived/queries/<dataset>/<category>/<name>.png    query images
    data/derived/catalog/queries.csv, meshes.csv            catalog (the rows of this dataset replaced)
    data/derived/catalog/<dataset>_queries_report.json      counts, failures, settings
    data/splits/<dataset>/<split name>.json                 split files

A mini run prepares only 5 photos per category and writes all outputs to
test_outputs/make_queries_mini/ instead, so that the real outputs are not touched.

Only Pix3D is supported now. ABO is added when its photos and masks exist: it needs one more
function that lists its photos and meshes (like `_pix3d_items`); the rest of the step is shared.
"""

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src import paths
from src.data import catalog, pix3d, splits
from src.preprocess.queries import FAILED, QueryJob, make_query_images

DATASETS = ("pix3d",)
MINI_PHOTOS_PER_CATEGORY = 5


@dataclass
class DatasetItems:
    """What the query step needs from one dataset."""

    photos: list  # one record per photo (fields as in pix3d.PhotoRecord)
    meshes: list  # one record per mesh entry (fields as in pix3d.MeshRecord)
    photos_listed: int  # number of photos in the dataset annotations, before any selection
    fixed_test: list[str] | None  # query ids of a fixed seen_meshes test part, or None


def output_dirs(mini: bool) -> dict[str, Path]:
    """Return the output folders: the real ones, or a separate test folder for a mini run."""
    if mini:
        base = paths.TEST_OUTPUTS_DIR / "make_queries_mini"
        return {"queries": base / "queries", "catalog": base / "catalog", "splits": base / "splits"}
    return {"queries": paths.QUERIES_DIR, "catalog": paths.CATALOG_DIR, "splits": paths.SPLITS_DIR}


def _first_per_category(photos: list, count: int) -> list:
    """Keep the first `count` photos of each category, in their original order."""
    taken: Counter = Counter()
    kept = []
    for photo in photos:
        taken[photo.category] += 1
        if taken[photo.category] <= count:
            kept.append(photo)
    return kept


def _pix3d_items(data_cfg: dict, mini: bool) -> DatasetItems:
    """List the Pix3D photos and meshes, and the v1 test part."""
    annotations = pix3d.load_annotations()
    photos = pix3d.photo_records(annotations)
    if mini:
        photos = _first_per_category(photos, MINI_PHOTOS_PER_CATEGORY)
    settings = data_cfg["pix3d"]
    fixed_test = (
        pix3d.v1_test_query_ids(annotations, settings["v1_seed"], settings["v1_holdout_fraction"])
        if settings["v1_test_part"]
        else None
    )
    return DatasetItems(photos, pix3d.mesh_records(annotations), len(annotations), fixed_test)


_ITEM_FUNCTIONS = {"pix3d": _pix3d_items}


def _make_splits(dataset: str, query_rows: list[dict], split_cfg: dict, fixed_test: list[str] | None) -> list[dict]:
    """Make the three splits of one dataset from its query rows."""
    seed = split_cfg["seed"]
    seen = split_cfg["seen_meshes"]
    made = [
        splits.seen_meshes_split(query_rows, seed, seen["test_fraction"], seen["val_fraction"], fixed_test),
        splits.unseen_meshes_split(query_rows, seed, split_cfg["unseen_meshes"]["fractions"]),
    ]
    unseen_categories = split_cfg["unseen_categories"]
    present = {row["category"] for row in query_rows}
    held_out = [c for c in unseen_categories["held_out"].get(dataset, []) if c in present]
    if held_out:  # a mini run can miss a held-out category; the split is then left out
        made.append(splits.unseen_categories_split(query_rows, held_out, seed, unseen_categories["val_fraction"]))
    return made


def run(dataset: str, cfg: dict, mini: bool = False, overwrite: bool = False, workers: int | None = None) -> dict:
    """Run the query step for one dataset, and return its report (also saved as JSON).

    Args:
        dataset: Name of the dataset, one of DATASETS.
        cfg: The merged settings (see `src/config.py`); the step reads `cfg["data"]`.
        mini: True = only 5 photos per category, outputs in test_outputs/make_queries_mini/.
        overwrite: True = make the query images again if they exist. Otherwise existing images
            are kept, and only the catalog rows and the splits are written again.
        workers: Number of parallel worker processes (None = the default, 8 or fewer).
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    data_cfg = cfg["data"]
    out = output_dirs(mini)
    start = time.perf_counter()

    # 1. The photos and meshes of the dataset.
    items = _ITEM_FUNCTIONS[dataset](data_cfg, mini)
    print(f"{dataset}: {len(items.photos)} photos to prepare{' (mini)' if mini else ''}")

    # 2. The query images, made in parallel.
    jobs = [
        QueryJob(
            query_id=p.query_id,
            photo_path=p.photo_path,
            mask_path=p.mask_path,
            output_path=str(out["queries"] / dataset / p.category / f"{Path(p.photo_path).stem}.png"),
            background_rgb=tuple(data_cfg["background_rgb"]),
            margin=data_cfg["crop_margin"],
            output_size=data_cfg["image_size"],
            overwrite=overwrite,
        )
        for p in items.photos
    ]
    results = make_query_images(jobs, workers)
    status = Counter(r["status"] for r in results)
    failed = [r for r in results if r["status"] == FAILED]
    failed_ids = {r["query_id"] for r in failed}
    print(f"query images: {status['done']} made, {status['skipped']} already there, {status['failed']} failed")
    for r in failed:
        print(f"  left out: {r['query_id']}  ({r['message']})")

    # 3. Catalog rows: the prepared queries, and the meshes that have at least one query.
    output_of = {job.query_id: Path(job.output_path) for job in jobs}
    query_rows = [
        {
            "query_id": p.query_id,
            "dataset": dataset,
            "category": p.category,
            "mesh_id": p.mesh_id,
            "photo_path": p.photo_path,
            "mask_path": p.mask_path,
            "query_path": str(output_of[p.query_id].relative_to(paths.ROOT)),
            "truncated": p.truncated,
            "occluded": p.occluded,
            "slightly_occluded": p.slightly_occluded,
        }
        for p in items.photos
        if p.query_id not in failed_ids
    ]
    photos_per_mesh = Counter(row["mesh_id"] for row in query_rows)
    mesh_rows = [
        {
            "mesh_id": m.mesh_id,
            "dataset": dataset,
            "category": m.category,
            "mesh_path": m.mesh_path,
            "render_dir": str((paths.RENDERS_DIR / m.mesh_id).relative_to(paths.ROOT)),
            "num_photos": photos_per_mesh[m.mesh_id],
            "num_mesh_variants": m.num_mesh_variants,
        }
        for m in items.meshes
        if photos_per_mesh[m.mesh_id] > 0
    ]
    catalog.replace_dataset_rows(out["catalog"] / paths.QUERIES_CSV.name, query_rows, dataset, catalog.QUERY_COLUMNS)
    catalog.replace_dataset_rows(out["catalog"] / paths.MESHES_CSV.name, mesh_rows, dataset, catalog.MESH_COLUMNS)
    print(f"catalog: {len(query_rows)} query rows, {len(mesh_rows)} mesh rows -> {out['catalog'].relative_to(paths.ROOT)}")

    # 4. Split files.
    made_splits = _make_splits(dataset, query_rows, data_cfg["splits"], items.fixed_test)
    for split in made_splits:
        split_path = splits.save_split(split, paths.split_file(dataset, split["name"], out["splits"]))
        print(f"split {split['name']}: {split['counts']} -> {split_path.relative_to(paths.ROOT)}")

    # 5. Report: counts, failures, settings, and run time.
    report = {
        "dataset": dataset,
        "mini": mini,
        "photos_listed": items.photos_listed,
        "queries_prepared": len(query_rows),
        "meshes": len(mesh_rows),
        "status": dict(status),
        "left_out": failed,
        "queries_per_category": dict(sorted(Counter(r["category"] for r in query_rows).items())),
        "settings": {k: data_cfg[k] for k in ("image_size", "background_rgb", "crop_margin")},
        "splits": {split["name"]: split["counts"] for split in made_splits},
        "seconds": round(time.perf_counter() - start, 1),
    }
    report_path = paths.queries_report_file(dataset, out["catalog"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {report_path.relative_to(paths.ROOT)} ({report['seconds']} s)")
    return report

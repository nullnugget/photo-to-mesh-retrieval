"""Query images at another side length than the main one (`data.image_size`), for example 448 pixels.

The patch comparison of the re-ranking sees more detail in larger images (the backbone takes any
multiple of its patch size). This step makes the query images of every query set again, at the given
size, from the original photos and the same masks, with the same crop function (`crop_to_square`):
the same crop, only the last resize differs. The catalog, the splits, and the images of the main size
are not touched.

The photos of a set are the rows of its feature file (`05_extract_features.py` must be complete), so
the images of both sizes cover the same photos:

| Set (dataset)             | Mask                                     | Folder (448 pixels)                          |
|---------------------------|------------------------------------------|----------------------------------------------|
| queries_reference (pix3d) | the catalog mask (Pix3D's own masks)      | `data/derived/queries_448/pix3d/...`          |
| queries_sam (pix3d)       | the SAM mask of the segment step          | `data/derived/masks/pix3d_sam_queries_448/...` |
| queries_sam (abo)         | the catalog mask (the SAM masks of ABO)   | `data/derived/queries_448/abo/...`            |

Existing images are skipped, unless overwrite is given. The renders of another size are made by the
render step (`04_render.py --set render.output_size=448`, into `data/derived/renders_448/`).

A mini run makes the first 16 images of each set into `test_outputs/sized_queries_mini/`.

Outputs (normal run): the images above, and `data/derived/queries_<size>/sized_queries_report.json`.
"""

import json
import time
from collections import Counter
from pathlib import Path

from src import paths
from src.data.catalog import read_table
from src.models.feature_store import read_features
from src.pipeline.extract_features import (CATALOG_IMAGES, QUERY_SETS, catalog_query_file, sam_mask_file,
                                           sam_query_file)
from src.preprocess.queries import DONE, FAILED, SKIPPED, QueryJob, make_query_images

MINI_IMAGES = 16


def target_file(dataset: str, query_set: str, row: dict, suffix: str, mini: bool) -> Path:
    """Return the file of one query image at the size of `suffix` (the mini run: below test_outputs/)."""
    if QUERY_SETS[dataset][query_set] == CATALOG_IMAGES:
        target = catalog_query_file(row, suffix)
    else:
        target = sam_query_file(paths.MASKS_DIR, row, suffix)
    if mini:
        return paths.TEST_OUTPUTS_DIR / "sized_queries_mini" / target.relative_to(paths.DERIVED_DIR)
    return target


def run(cfg: dict, size: int, datasets: list[str], mini: bool = False, overwrite: bool = False,
        workers: int | None = None) -> dict:
    """Make the query images of every query set of `datasets` at `size` pixels; return the report (also saved).

    Raises:
        ValueError: If `size` is the main size (those images are made by the query and feature steps).
    """
    data_cfg, backbone = cfg["data"], cfg["backbone"]["name"]
    suffix = paths.size_suffix(size, data_cfg["image_size"])
    if not suffix:
        raise ValueError(f"{size} is the main image size; its query images are made by 03_make_queries.py and 05_extract_features.py")
    report = {"size": size, "mini": mini, "sets": {}}
    for dataset in datasets:
        by_id = {q["query_id"]: q for q in read_table(paths.QUERIES_CSV) if q["dataset"] == dataset}
        for query_set, source in QUERY_SETS[dataset].items():
            ids = read_features(paths.features_file(backbone, dataset, query_set)).ids
            rows = [by_id[q] for q in ids[:MINI_IMAGES if mini else None]]
            jobs = [QueryJob(
                query_id=row["query_id"],
                photo_path=row["photo_path"],
                mask_path=row["mask_path"] if source == CATALOG_IMAGES else str(sam_mask_file(row)),
                output_path=str(target_file(dataset, query_set, row, suffix, mini)),
                background_rgb=tuple(data_cfg["background_rgb"]),
                margin=data_cfg["crop_margin"],
                output_size=size,
                overwrite=overwrite,
            ) for row in rows]
            start = time.perf_counter()
            results = make_query_images(jobs, workers)
            status = Counter(r["status"] for r in results)
            report["sets"][f"{dataset}_{query_set}"] = {
                "images": len(jobs), "made": status[DONE], "skipped": status[SKIPPED],
                "failed": [r for r in results if r["status"] == FAILED],
                "seconds": round(time.perf_counter() - start, 1)}
            print(f"{dataset}_{query_set}: {status[DONE]} made, {status[SKIPPED]} already there, {status[FAILED]} failed")
    base = paths.TEST_OUTPUTS_DIR / "sized_queries_mini" if mini else paths.DERIVED_DIR
    target = paths.sized(base / "queries", suffix) / "sized_queries_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {target.relative_to(paths.ROOT)}")
    return report

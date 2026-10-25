"""The query step: query images, catalog rows, and split files for one dataset.

For every photo of the dataset, this step:

1. reads the photo and its object mask;
2. makes the query image with the shared crop function (grey background, tight square crop,
   224 x 224), and saves it as PNG;
3. writes one row per query and one row per mesh into the shared catalog;
4. writes the three split files (seen_meshes, unseen_meshes, unseen_categories).

Photos that cannot be prepared (in Pix3D: two photos whose mask belongs to another image) are
reported and left out of the catalog and the splits.

Where the photos, masks, and meshes come from:

- Pix3D: `pix3d.json`; the masks come with the dataset; one mesh row per model folder, except that
  the twin groups of the merge stage (`10_find_twins.py --dataset pix3d --stage merge`) are one row
  each: their photos show the group's representative and take the group's class (`merge_twins`).
- ABO: the photos that the segment step kept (`data/derived/masks/abo_segment_results.csv`, verdict
  "kept") and the photo check did not leave out (`check_photos`: an extra image that shows only a part,
  a drawing, or a swatch), with their SAM masks. Each photo's mesh is the gallery entry of its geometry group
  (`catalog_abo_meshes.photo_entry`), so that colour variants of one geometry are one answer. The mesh
  rows are the ABO rows that the catalog stage of the download step wrote, all of them: a mesh without
  a kept photo stays in the gallery (a distractor that no photo asks for, like a new product), and its
  `num_photos` is 0. ABO photos have no truncated or occluded flags (the cells stay empty).

Outputs (normal run):
    data/derived/queries/<dataset>/<category>/<name>.png    query images
    data/derived/catalog/queries.csv, meshes.csv            catalog (the rows of this dataset replaced)
    data/derived/catalog/<dataset>_queries_report.json      counts, failures, settings
    data/splits/<dataset>/<split name>.json                 split files

A mini run prepares only 5 photos per category and writes all outputs to
test_outputs/make_queries_mini/ instead, so that the real outputs are not touched. For ABO, the mini
run takes its photos from the mini outputs of the segment step (`02_segment_photos.py --mini`).
"""

import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from src import paths
from src.data import abo, catalog, pix3d, splits
from src.pipeline import catalog_abo_meshes, check_photos, download_abo, find_twins, segment_photos
from src.preprocess.queries import FAILED, QueryJob, make_query_images

DATASETS = ("pix3d", "abo")
MINI_PHOTOS_PER_CATEGORY = 5


@dataclass(frozen=True)
class QueryPhoto:
    """One photo for the query step (the fields that the step reads; `pix3d.PhotoRecord` has them too)."""

    query_id: str
    category: str
    mesh_id: str  # the gallery entry that the photo shows
    photo_path: str  # relative to the repository root
    mask_path: str  # relative to the repository root
    truncated: bool | None  # None = unknown (ABO)
    occluded: bool | None
    slightly_occluded: bool | None


@dataclass(frozen=True)
class GalleryMesh:
    """One gallery entry for the query step (the fields that the step reads; `pix3d.MeshRecord` has them too)."""

    mesh_id: str
    category: str
    mesh_path: str
    num_mesh_variants: int


@dataclass
class DatasetItems:
    """What the query step needs from one dataset."""

    photos: list  # one record per photo (fields as in QueryPhoto)
    meshes: list  # one record per mesh entry (fields as in GalleryMesh)
    photos_listed: int  # number of photos in the dataset annotations, before any selection
    fixed_test: list[str] | None  # query ids of a fixed seen_meshes test part, or None
    keep_meshes_without_photos: bool = False  # True = every mesh gets a catalog row, also with 0 photos


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


def merge_twins(photos: list, meshes: list, twins: dict[str, tuple[str, str]]) -> tuple[list, list]:
    """Apply the twin groups of the merge stage (`find_twins.read_twin_groups`): each photo of a group
    entry shows the group's representative and takes the group's class; the mesh list keeps one entry per
    group (the representative, with the group's class and the mesh files of all its entries).

    The query ids stay as they are (in Pix3D the photo names repeat across categories, so the category
    part of the id keeps the photo's own folder).
    """
    if not twins:
        return photos, meshes
    photos = [replace(p, mesh_id=twins[p.mesh_id][0], category=twins[p.mesh_id][1]) if p.mesh_id in twins else p
              for p in photos]
    variants = Counter()
    for m in meshes:
        variants[twins[m.mesh_id][0] if m.mesh_id in twins else m.mesh_id] += m.num_mesh_variants
    kept = []
    for m in meshes:
        if m.mesh_id in twins:
            representative, category = twins[m.mesh_id]
            if m.mesh_id != representative:
                continue
            m = replace(m, category=category)
        kept.append(replace(m, num_mesh_variants=variants[m.mesh_id]))
    return photos, kept


def _pix3d_items(cfg: dict, mini: bool) -> DatasetItems:
    """List the Pix3D photos and meshes (twin groups merged, `merge_twins`), and the v1 test part."""
    data_cfg = cfg["data"]
    annotations = pix3d.load_annotations()
    photos, meshes = merge_twins(pix3d.photo_records(annotations), pix3d.mesh_records(annotations),
                                 find_twins.read_twin_groups(pix3d.DATASET))
    if mini:
        photos = _first_per_category(photos, MINI_PHOTOS_PER_CATEGORY)
    settings = data_cfg["pix3d"]
    fixed_test = (
        pix3d.v1_test_query_ids(annotations, settings["v1_seed"], settings["v1_holdout_fraction"])
        if settings["v1_test_part"]
        else None
    )
    return DatasetItems(photos, meshes, len(annotations), fixed_test)


def _abo_items(cfg: dict, mini: bool) -> DatasetItems:
    """List the ABO photos that the segment step kept and the photo check did not leave out (an extra image
    above `segment.cleaning.max_bad_score`, `check_photos`), their gallery entries, and all ABO meshes."""
    _, photos = abo.read_manifest(paths.ABO_MANIFEST_DIR)
    groups = catalog_abo_meshes.read_geometry_groups(paths.ABO_MANIFEST_DIR)
    by_id = {p.image_id: p for p in photos}
    meshes = [m for m in catalog.read_table(paths.MESHES_CSV) if m["dataset"] == abo.DATASET]
    category_of = {m["mesh_id"]: m["category"] for m in meshes}

    masks_root = segment_photos.output_root(mini)
    results = catalog.read_table(segment_photos.results_file(abo.DATASET, masks_root))
    if check_photos.results_file().exists():
        checked = check_photos.read_results()
    elif mini:
        checked = {}
        print("no photo check results: a mini run keeps every photo of the segment step")
    else:
        raise FileNotFoundError(f"{check_photos.results_file()} is missing: run scripts/11_check_photos.py first")
    max_bad_score = cfg["segment"]["cleaning"]["max_bad_score"]
    records, left_out = [], 0
    for row in results:
        if row["verdict"] != segment_photos.KEPT:
            continue
        if row["photo_id"] in checked and check_photos.is_left_out(*checked[row["photo_id"]], max_bad_score):
            left_out += 1
            continue
        photo = by_id[row["photo_id"].split("/")[-1]]
        entry = catalog_abo_meshes.photo_entry(photo, groups)  # never None: the segment step skipped those photos
        item = segment_photos.PhotoItem(row["photo_id"], photo.category, "", None, f"{photo.image_id}.png")
        records.append(QueryPhoto(
            query_id=f"{abo.DATASET}/{category_of[entry]}/{photo.image_id}",
            category=category_of[entry],
            mesh_id=entry,
            photo_path=str(download_abo.photo_file(paths.ABO_IMAGES_DIR, photo).relative_to(paths.ROOT)),
            mask_path=str(segment_photos.mask_file(masks_root, abo.DATASET, item).relative_to(paths.ROOT)),
            truncated=None, occluded=None, slightly_occluded=None,
        ))
    records.sort(key=lambda r: r.query_id)
    print(f"photo check: {left_out} extra images left out (bad score above {max_bad_score})")
    if mini:
        records = _first_per_category(records, MINI_PHOTOS_PER_CATEGORY)
    gallery = [GalleryMesh(m["mesh_id"], m["category"], m["mesh_path"], int(m["num_mesh_variants"])) for m in meshes]
    return DatasetItems(records, gallery, len(photos), fixed_test=None, keep_meshes_without_photos=True)


_ITEM_FUNCTIONS = {"pix3d": _pix3d_items, "abo": _abo_items}


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


def _keep_parts(dataset: str, fresh: dict, query_rows: list[dict], data_cfg: dict, splits_dir: Path) -> dict:
    """The fresh split with the parts of the existing split file carried over (`splits.carry_over_parts`)."""
    path = paths.split_file(dataset, fresh["name"], splits_dir)
    if not path.exists():
        print(f"split {fresh['name']}: no earlier file, new parts")
        return fresh
    old = splits.load_split(path)
    # The photo key: ABO query ids can change their category part (the image id stays); Pix3D query ids
    # do not change, and Pix3D photo names repeat across categories, so the whole id is the key.
    key = (lambda query_id: query_id) if dataset == pix3d.DATASET else (lambda query_id: query_id.rsplit("/", 1)[-1])
    kept = splits.carry_over_parts(old, fresh, query_rows, by_mesh=fresh["name"] == "unseen_meshes", photo_key=key)
    if fresh["name"] == "unseen_categories":  # the test part must stay exactly the held-out categories
        held = set(data_cfg["splits"]["unseen_categories"]["held_out"].get(dataset, []))
        category = {r["query_id"]: r["category"] for r in query_rows}
        wrong = [q for p, ids in kept["parts"].items() for q in ids if (category[q] in held) != (p == "test")]
        if wrong:
            raise ValueError(f"unseen_categories: {len(wrong)} photos are in the wrong part for their category after the change")
    print(f"split {fresh['name']}: parts carried over; earlier {old['counts']}, now {kept['counts']}")
    return kept


def run(dataset: str, cfg: dict, mini: bool = False, overwrite: bool = False, workers: int | None = None,
        keep_parts: bool = False) -> dict:
    """Run the query step for one dataset, and return its report (also saved as JSON).

    Args:
        dataset: Name of the dataset, one of DATASETS.
        cfg: The merged settings (see `src/config.py`); the step reads `cfg["data"]`.
        mini: True = only 5 photos per category, outputs in test_outputs/make_queries_mini/.
        overwrite: True = make the query images again if they exist. Otherwise existing images
            are kept, and only the catalog rows and the splits are written again.
        workers: Number of parallel worker processes (None = the default, 8 or fewer).
        keep_parts: True = each photo keeps its part of the existing split files (after a catalog
            change, for example merged ABO entries; `splits.carry_over_parts`), so that no photo moves
            into a test part. False = new splits from the seed.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    data_cfg = cfg["data"]
    out = output_dirs(mini)
    start = time.perf_counter()

    # 1. The photos and meshes of the dataset.
    items = _ITEM_FUNCTIONS[dataset](cfg, mini)
    print(f"{dataset}: {len(items.photos)} photos to prepare{' (mini)' if mini else ''}")

    # 2. The query images, made in parallel.
    jobs = [
        QueryJob(
            query_id=p.query_id,
            photo_path=p.photo_path,
            mask_path=p.mask_path,
            output_path=str(out["queries"] / f"{p.query_id}.png"),  # "<dataset>/<category>/<name>": unique, also after a merge
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
        if photos_per_mesh[m.mesh_id] > 0 or items.keep_meshes_without_photos
    ]
    catalog.replace_dataset_rows(out["catalog"] / paths.QUERIES_CSV.name, query_rows, dataset, catalog.QUERY_COLUMNS)
    catalog.replace_dataset_rows(out["catalog"] / paths.MESHES_CSV.name, mesh_rows, dataset, catalog.MESH_COLUMNS)
    print(f"catalog: {len(query_rows)} query rows, {len(mesh_rows)} mesh rows -> {out['catalog'].relative_to(paths.ROOT)}")

    # 4. Split files.
    made_splits = _make_splits(dataset, query_rows, data_cfg["splits"], items.fixed_test)
    if keep_parts:
        made_splits = [_keep_parts(dataset, split, query_rows, data_cfg, out["splits"]) for split in made_splits]
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

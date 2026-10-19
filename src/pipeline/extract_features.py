"""The feature step: backbone features of the renders and the query images of one dataset.

For one dataset and one backbone, this step writes one feature file per feature set
(`src/models/feature_store.py` describes the file content):

| Feature set       | Images                                                   | Datasets     |
|-------------------|----------------------------------------------------------|--------------|
| renders           | the 25 views of every gallery mesh                        | pix3d, abo   |
| queries_reference | the catalog's query images, cut out with Pix3D's masks    | pix3d        |
| queries_sam       | query images cut out with the SAM masks of the segment step | pix3d, abo |

- The query sets hold every query image of the dataset, not only one split: the three splits put
  the same photo into different parts, and a split file selects its rows by query_id.
- ABO has no masks of its own, so its catalog query images are already the SAM ones.
- Pix3D's SAM query images are made here, with the same crop function as the query step, into
  `data/derived/masks/pix3d_sam_queries/` (the folder that the segment check notebook also uses).
  An existing image is made again when its mask file is newer.
- Every image gets both basic parts ("cls" and "masked_mean"); the object mask of the masked mean
  is every pixel that is not the background grey.

A set whose file exists is skipped, unless overwrite is given. The report
(`<backbone>/<dataset>_features_report.json`) keeps the entries of the sets of earlier runs.

A mini run takes 2 meshes per category and 3 photos of each, and writes all outputs to
`test_outputs/extract_features_mini/`, so that the real outputs are not touched.

Outputs (normal run):
    data/derived/features/<backbone>/<dataset>_<set>.npz                 the feature files
    data/derived/features/<backbone>/<dataset>_features_report.json      rows, speed, memory, left-out images
    data/derived/masks/pix3d_sam_queries/<category>/<name>.png           Pix3D query images with SAM masks
"""

import datetime
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import paths
from src.data.catalog import read_table
from src.models.backbones import FEATURE_PARTS, FeatureExtractor, object_masks
from src.models.feature_store import QUERIES, RENDERS, FeatureTable, write_features
from src.pipeline import segment_photos
from src.pipeline.render_meshes import VIEW_NAME, render_folder
from src.preprocess.queries import FAILED, QueryJob, make_query_images
from src.render.rig import rig_from_config
from src.runs import git_state

DATASETS = ("pix3d", "abo")
CATALOG_IMAGES, SAM_IMAGES = "catalog", "sam"  # where the query images of a set come from
# The query sets of each dataset, and the source of their images.
QUERY_SETS = {
    "pix3d": {"queries_reference": CATALOG_IMAGES, "queries_sam": SAM_IMAGES},
    "abo": {"queries_sam": CATALOG_IMAGES},
}
MINI_MESHES_PER_CATEGORY = 2
MINI_PHOTOS_PER_MESH = 3


def feature_sets(dataset: str) -> list[str]:
    """Return the feature sets of one dataset: renders first, then its query sets."""
    return [RENDERS, *QUERY_SETS[dataset]]


def output_root(mini: bool) -> Path:
    """Return the folder of the feature files: the real one, or a test folder for a mini run."""
    return paths.TEST_OUTPUTS_DIR / "extract_features_mini" / "features" if mini else paths.FEATURES_DIR


def sam_queries_root(mini: bool) -> Path:
    """Return the folder that holds the `<dataset>_sam_queries/` folders."""
    return paths.TEST_OUTPUTS_DIR / "extract_features_mini" / "masks" if mini else paths.MASKS_DIR


def sam_query_file(root: Path, query_row: dict) -> Path:
    """Return the query image made with the SAM mask, e.g. <root>/pix3d_sam_queries/chair/0001.png."""
    return root / f"{query_row['dataset']}_sam_queries" / query_row["category"] / Path(query_row["query_path"]).name


def sam_mask_file(query_row: dict) -> Path:
    """Return the SAM mask of a query's photo, as the segment step names it."""
    item = segment_photos.PhotoItem(query_row["query_id"], query_row["category"], query_row["photo_path"], None,
                                    Path(query_row["photo_path"]).stem + ".png")
    return segment_photos.mask_file(paths.MASKS_DIR, query_row["dataset"], item)


def report_file(backbone: str, dataset: str, root: Path = paths.FEATURES_DIR) -> Path:
    return root / backbone / f"{dataset}_features_report.json"


def catalog_rows(dataset: str, mini: bool) -> tuple[list[dict], list[dict]]:
    """Return the mesh rows and the query rows of one dataset (for a mini run, a small sample).

    The mini sample: the first 2 meshes with photos of each category, and the first 3 photos of
    each of these meshes, so that every mini query has its mesh in the mini gallery.
    """
    meshes = [m for m in read_table(paths.MESHES_CSV) if m["dataset"] == dataset]
    queries = [q for q in read_table(paths.QUERIES_CSV) if q["dataset"] == dataset]
    if not meshes or not queries:
        raise ValueError(f"the catalog has no mesh rows or no query rows of dataset {dataset!r}")
    if mini:
        taken: Counter = Counter()
        meshes = [m for m in meshes if m["num_photos"] > 0 and (taken.update([m["category"]]) or taken[m["category"]] <= MINI_MESHES_PER_CATEGORY)]
        kept = {m["mesh_id"] for m in meshes}
        per_mesh: Counter = Counter()
        queries = [q for q in queries if q["mesh_id"] in kept and (per_mesh.update([q["mesh_id"]]) or per_mesh[q["mesh_id"]] <= MINI_PHOTOS_PER_MESH)]
    return meshes, queries


# ---------------------------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------------------------


class _ImageFiles(Dataset):
    """Reads image files as uint8 RGB arrays. The data loader reads them in parallel worker processes."""

    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> np.ndarray:
        return np.asarray(Image.open(self.files[index]).convert("RGB"))


def render_files(meshes: list[dict], views: int) -> list[Path]:
    """Return the view files of all meshes, mesh by mesh, in view order.

    Raises:
        FileNotFoundError: If a view is missing (the render step must be complete first).
    """
    files = [render_folder(m, paths.RENDERS_DIR) / VIEW_NAME.format(v) for m in meshes for v in range(views)]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} render views are missing, for example {missing[0]}; run the render step first")
    return files


def make_sam_query_images(queries: list[dict], root: Path, data_cfg: dict, overwrite: bool) -> tuple[list[dict], list[dict]]:
    """Make the query images with the SAM masks; return (rows with an image, left-out rows with a reason).

    An image is made again when `overwrite` is True or when its mask is newer than the image (the
    masks were made again after the image). A photo without a SAM mask is left out.
    """
    jobs, left_out = [], []
    for row in queries:
        mask, target = sam_mask_file(row), sam_query_file(root, row)
        if not mask.exists():
            left_out.append({"query_id": row["query_id"], "message": "no SAM mask"})
            continue
        stale = target.exists() and target.stat().st_mtime < mask.stat().st_mtime
        jobs.append(QueryJob(
            query_id=row["query_id"],
            photo_path=row["photo_path"],
            mask_path=str(mask),
            output_path=str(target),
            background_rgb=tuple(data_cfg["background_rgb"]),
            margin=data_cfg["crop_margin"],
            output_size=data_cfg["image_size"],
            overwrite=overwrite or stale,
        ))
    results = make_query_images(jobs)  # one result per job, in the same order
    status = Counter(r["status"] for r in results)
    print(f"SAM query images: {status['done']} made, {status['skipped']} already there, {status['failed']} failed")
    failed = {r["query_id"]: r["message"] for r in results if r["status"] == FAILED}
    left_out += [{"query_id": query_id, "message": message} for query_id, message in failed.items()]
    by_id = {row["query_id"]: row for row in queries}
    return [by_id[job.query_id] for job in jobs if job.query_id not in failed], left_out


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------


def extract_files(extractor: FeatureExtractor, files: list[Path], background_rgb: tuple, workers: int, desc: str) -> dict[str, np.ndarray]:
    """Return both basic parts of every image file, in file order: {part: float16 (N, dim)}.

    Worker processes read the files while the GPU works on the previous batch.
    """
    loader = DataLoader(_ImageFiles(files), batch_size=extractor.batch_size, num_workers=workers, collate_fn=np.stack)
    outputs: dict[str, list] = {part: [] for part in FEATURE_PARTS}
    for images in tqdm(loader, desc=desc, unit="batch"):
        parts = extractor.extract_parts(images, object_masks(images, background_rgb))
        for part, vectors in parts.items():
            outputs[part].append(vectors.astype(np.float16))
    return {part: np.concatenate(vectors) for part, vectors in outputs.items()}


def repeat_difference(extractor: FeatureExtractor, files: list[Path], background_rgb: tuple) -> float:
    """Extract the first batch twice and return the largest difference of one number (expected about 0)."""
    images = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files[: extractor.batch_size]])
    masks = object_masks(images, background_rgb)
    first, second = extractor.extract_parts(images, masks), extractor.extract_parts(images, masks)
    return float(max(np.abs(first[part] - second[part]).max() for part in FEATURE_PARTS))


def _file_info(extractor: FeatureExtractor, cfg: dict, dataset: str, feature_set: str, source: str, mini: bool) -> dict:
    """The record of how a feature file was made (saved inside the file)."""
    commit, dirty = git_state()
    info = {
        "backbone": extractor.spec.name,
        "weights": extractor.spec.weights,
        "dataset": dataset,
        "feature_set": feature_set,
        "images": source,
        "object_mask": "pixels that differ from the background grey",
        "image_size": extractor.image_size,
        "precision": cfg["backbone"]["precision"],
        "background_rgb": list(cfg["data"]["background_rgb"]),
        "mini": mini,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
    }
    if feature_set == RENDERS:
        info.update(lighting=cfg["render"]["lighting"], rig=cfg["render"]["rig"])
    return info


def run(dataset: str, cfg: dict, backbone: str | None = None, sets: list[str] | None = None,
        mini: bool = False, overwrite: bool = False, device: str = "cuda") -> dict:
    """Write the feature files of one dataset, and return the report (also saved as JSON).

    Args:
        dataset: One of DATASETS.
        cfg: The merged settings; the step reads `backbone`, `data`, and `render`.
        backbone: Backbone name (None = `backbone.name`).
        sets: Feature sets to write (None = all sets of the dataset, see `feature_sets`).
        mini: True = a small sample, outputs in test_outputs/extract_features_mini/.
        overwrite: True = write a set again if its file exists (and make the SAM query images again).
        device: "cuda" or "cpu".
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    backbone = backbone or cfg["backbone"]["name"]
    sets = sets or feature_sets(dataset)
    unknown = [s for s in sets if s not in feature_sets(dataset)]
    if unknown:
        raise ValueError(f"{dataset} has no feature sets {unknown}; it has {feature_sets(dataset)}")
    root = output_root(mini)
    background = tuple(cfg["data"]["background_rgb"])
    workers = cfg["backbone"]["loader_workers"]
    meshes, queries = catalog_rows(dataset, mini)
    to_write = [s for s in sets if overwrite or not paths.features_file(backbone, dataset, s, root).exists()]
    print(f"{dataset}: {len(meshes)} meshes, {len(queries)} query photos{' (mini)' if mini else ''}; backbone {backbone}; "
          f"sets to write {to_write}, skipped (file exists) {[s for s in sets if s not in to_write]}")

    entries = {}
    extractor = FeatureExtractor(cfg["backbone"], name=backbone, device=device) if to_write else None
    for feature_set in to_write:
        left_out: list[dict] = []
        if feature_set == RENDERS:
            views = len(rig_from_config(cfg["render"]))
            files = render_files(meshes, views)
            source = f"render views ({views} per mesh)"
        else:
            rows = queries
            if QUERY_SETS[dataset][feature_set] == SAM_IMAGES:
                rows, left_out = make_sam_query_images(queries, sam_queries_root(mini), cfg["data"], overwrite)
                files = [sam_query_file(sam_queries_root(mini), row) for row in rows]
                source = "query images cut out with the SAM masks"
            else:
                files = [paths.resolve(row["query_path"]) for row in rows]
                source = "catalog query images" + (" (Pix3D masks)" if dataset == "pix3d" else " (SAM masks)")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        parts = extract_files(extractor, files, background, workers, desc=feature_set)
        seconds = time.perf_counter() - start

        if feature_set == RENDERS:
            parts = {part: vectors.reshape(len(meshes), views, -1) for part, vectors in parts.items()}
            labels = {"mesh_id": [m["mesh_id"] for m in meshes], "category": [m["category"] for m in meshes]}
        else:
            labels = {name: [row[name] for row in rows] for name in ("query_id", "mesh_id", "category")}
        info = _file_info(extractor, cfg, dataset, feature_set, source, mini)
        table = FeatureTable(RENDERS if feature_set == RENDERS else QUERIES, parts,
                             {name: np.asarray(values) for name, values in labels.items()}, info)
        target = write_features(table, paths.features_file(backbone, dataset, feature_set, root))
        entries[feature_set] = {
            "file": str(target.relative_to(paths.ROOT)),
            "rows": table.rows,
            "images": len(files),
            "left_out": left_out,
            "size_mb": round(target.stat().st_size / 2**20, 1),
            "seconds": round(seconds, 1),
            "images_per_second": round(len(files) / seconds, 1),
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
            "repeat_max_difference": repeat_difference(extractor, files, background),
            "created": info["created"],
        }
        print(f"{feature_set}: {table.rows} rows, {len(files)} images, {len(left_out)} left out, "
              f"{entries[feature_set]['images_per_second']} images/s -> {entries[feature_set]['file']}")

    report_path = report_file(backbone, dataset, root)
    earlier = json.loads(report_path.read_text())["sets"] if report_path.exists() else {}
    report = {"dataset": dataset, "backbone": backbone, "mini": mini, "sets": {**earlier, **entries}}  # earlier sets stay
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {report_path.relative_to(paths.ROOT)}")
    return report

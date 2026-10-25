"""The patch step: the reduced patch vectors of every render and query image (the patch store).

Two stages (`src/models/patch_store.py` describes the store):

1. fit: the PCA of the patch vectors. `rerank.patches.fit_images` random render views and as many
   random query images, from non-test images only:
   - query images: photos in no test part of any split (all query sets of both datasets);
   - renders: meshes with no photo in the test part of unseen_meshes or unseen_categories (the
     meshes of the seen_meshes test photos are training meshes there, so they count as non-test).
   Only the object patches (object share >= `rerank.patches.object_share`) enter the PCA, because
   the grey background patches would take the largest directions.
2. extract: every set of both datasets, projected to `rerank.patches.dim` numbers and stored. The
   rows follow the feature file of the same set (the feature step must be complete), so the patch
   store and the feature store have the same rows in the same order.

The projection is fitted once. `extract` stops if it is missing; `fit` with overwrite fits it again
(the sets must then be extracted again too: the step removes no old sets by itself).

A mini run takes the first 4 meshes and 32 query images of each set and 64 fit images, into
`test_outputs/extract_patches_mini/`.

Outputs (normal run):
    data/derived/patches/<backbone>/projection.npz
    data/derived/patches/<backbone>/<dataset>_<set>.npy, _object.npy, _labels.npz
    data/derived/patches/<backbone>/patches_report.json
"""

import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.data.catalog import read_table
from src.models.backbones import FeatureExtractor, object_masks
from src.models.feature_store import QUERIES, RENDERS, read_features
from src.models.patch_store import CovarianceSum, PatchProjection, PatchWriter, patch_files, projection_file
from src.pipeline.extract_features import (CATALOG_IMAGES, DATASETS, QUERY_SETS, ImageFiles, catalog_query_file,
                                           feature_sets, render_files, sam_query_file)
from src.runs import git_state

MINI_MESHES, MINI_QUERIES, MINI_FIT = 4, 32, 64
FIT_SPLITS = {"photos": paths.SPLIT_NAMES, "meshes": ("unseen_meshes", "unseen_categories")}  # test parts that are left out


def output_root(mini: bool) -> Path:
    return paths.TEST_OUTPUTS_DIR / "extract_patches_mini" if mini else paths.PATCHES_DIR


def set_images(backbone: str, dataset: str, feature_set: str, mini: bool = False,
               suffix: str = "") -> tuple[str, dict, list[Path], int]:
    """Return (kind, labels, image files, views) of one set, in the row order of its feature file.

    For renders the files are mesh by mesh, view by view; views is 1 for query sets. `suffix`
    (`paths.size_suffix`): the files of the same images at another size (for example `renders_448/`).
    """
    table = read_features(paths.features_file(backbone, dataset, feature_set))
    labels = dict(table.labels)
    if mini:
        keep = MINI_MESHES if table.kind == RENDERS else MINI_QUERIES
        labels = {name: values[:keep] for name, values in labels.items()}
    if table.kind == RENDERS:
        views = table.parts["cls"].shape[1]
        by_id = {m["mesh_id"]: m for m in read_table(paths.MESHES_CSV) if m["dataset"] == dataset}
        return RENDERS, labels, render_files([by_id[m] for m in labels["mesh_id"]], views, paths.sized(paths.RENDERS_DIR, suffix)), views
    by_id = {q["query_id"]: q for q in read_table(paths.QUERIES_CSV) if q["dataset"] == dataset}
    rows = [by_id[q] for q in labels["query_id"]]
    if QUERY_SETS[dataset][feature_set] == CATALOG_IMAGES:
        files = [catalog_query_file(row, suffix) for row in rows]
    else:
        files = [sam_query_file(paths.MASKS_DIR, row, suffix) for row in rows]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} query images are missing, for example {missing[0]}")
    return QUERIES, labels, files, 1


def test_parts(dataset: str) -> tuple[set[str], set[str]]:
    """Return (query ids in any test part, mesh ids with a photo in the test part of a new-mesh split)."""
    mesh_of = {q["query_id"]: q["mesh_id"] for q in read_table(paths.QUERIES_CSV) if q["dataset"] == dataset}
    photos, meshes = set(), set()
    for split in paths.SPLIT_NAMES:
        test = json.loads(paths.split_file(dataset, split).read_text())["parts"]["test"]
        photos.update(test)
        if split in FIT_SPLITS["meshes"]:
            meshes.update(mesh_of[q] for q in test)
    return photos, meshes


def fit_images(backbone: str, count: int, seed: int) -> tuple[list[Path], list[Path]]:
    """Return (render files, query image files) for the PCA fit: `count` random non-test images of each kind."""
    renders, queries = [], []
    for dataset in DATASETS:
        test_photos, test_meshes = test_parts(dataset)
        for feature_set in feature_sets(dataset):
            kind, labels, files, views = set_images(backbone, dataset, feature_set)
            if kind == RENDERS:
                renders += [f for i, f in enumerate(files) if labels["mesh_id"][i // views] not in test_meshes]
            else:
                queries += [f for q, f in zip(labels["query_id"], files) if q not in test_photos]
    rng = random.Random(seed)
    return rng.sample(renders, min(count, len(renders))), rng.sample(queries, min(count, len(queries)))


def _loader(files: list[Path], extractor: FeatureExtractor, workers: int) -> DataLoader:
    return DataLoader(ImageFiles(files), batch_size=extractor.batch_size, num_workers=workers, collate_fn=np.stack)


def fit_projection(extractor: FeatureExtractor, files: list[Path], background: tuple, object_share: float,
                   workers: int) -> PatchProjection:
    """Fit the PCA on the object patches of the given images."""
    sums = CovarianceSum(extractor.spec.dim, extractor.device)
    for images in tqdm(_loader(files, extractor, workers), desc="fit", unit="batch"):
        tokens, shares = extractor.extract_patches(images, object_masks(images, background))
        sums.add(tokens[shares >= object_share])
    return sums.fit()


def extract_set(extractor: FeatureExtractor, projection: PatchProjection, files: list[Path], writer: PatchWriter,
                dim: int, background: tuple, workers: int, desc: str) -> None:
    for images in tqdm(_loader(files, extractor, workers), desc=desc, unit="batch"):
        tokens, shares = extractor.extract_patches(images, object_masks(images, background))
        writer.add(projection.project(tokens, dim).half().cpu().numpy(), shares.cpu().numpy())


def run(cfg: dict, stages: list[str], backbone: str | None = None, mini: bool = False, overwrite: bool = False,
        device: str = "cuda") -> dict:
    """Run the stages ("fit", "extract") and return the report (also saved as JSON)."""
    pcfg = cfg["rerank"]["patches"]
    backbone = backbone or cfg["backbone"]["name"]
    root = output_root(mini)
    background = tuple(cfg["data"]["background_rgb"])
    workers = cfg["backbone"]["loader_workers"]
    extractor = FeatureExtractor(cfg["backbone"], name=backbone, device=device)
    report_path = root / backbone / "patches_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"backbone": backbone, "sets": {}}
    commit, dirty = git_state()
    target = projection_file(backbone, root)

    if "fit" in stages:
        if target.exists() and not overwrite:
            print(f"fit: {target} exists (overwrite to fit again)")
        else:
            count = MINI_FIT if mini else pcfg["fit_images"]
            render_list, query_list = fit_images(backbone, count, pcfg["seed"])
            start = time.perf_counter()
            projection = fit_projection(extractor, render_list + query_list, background, pcfg["object_share"], workers)
            projection.save(target)
            report["projection"] = {
                "file": str(target.relative_to(paths.ROOT)), "renders": len(render_list), "queries": len(query_list),
                "object_patches": projection.count, "seconds": round(time.perf_counter() - start, 1),
                "explained": {d: round(projection.explained(d), 4) for d in (16, 32, 64, 128, 256)},
                "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "git_commit": commit, "git_dirty": dirty}
            print(f"fit: {projection.count} object patches; explained variance "
                  + ", ".join(f"{d}: {v:.3f}" for d, v in report["projection"]["explained"].items()))

    if "extract" in stages:
        if not target.exists():
            raise FileNotFoundError(f"{target} is missing; run the fit stage first")
        projection = PatchProjection.load(target)
        for dataset in DATASETS:
            for feature_set in feature_sets(dataset):
                files_out = patch_files(backbone, dataset, feature_set, root)
                if files_out["vectors"].exists() and not overwrite:
                    print(f"{dataset}_{feature_set}: exists, skipped")
                    continue
                kind, labels, files, views = set_images(backbone, dataset, feature_set, mini)
                shape = (len(labels["mesh_id"]), views) if kind == RENDERS else (len(labels["query_id"]),)
                patches = (extractor.image_size // extractor.spec.patch_size) ** 2
                writer = PatchWriter(files_out, shape, pcfg["dim"], patches)
                start = time.perf_counter()
                extract_set(extractor, projection, files, writer, pcfg["dim"], background, workers, f"{dataset}_{feature_set}")
                seconds = time.perf_counter() - start
                info = {"backbone": backbone, "dataset": dataset, "feature_set": feature_set, "dim": pcfg["dim"],
                        "projection": str(target.relative_to(paths.ROOT)), "projection_count": projection.count,
                        "image_size": extractor.image_size, "precision": cfg["backbone"]["precision"], "mini": mini,
                        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                        "git_commit": commit, "git_dirty": dirty}
                writer.close(kind, labels, info)
                report["sets"][f"{dataset}_{feature_set}"] = {
                    "rows": len(next(iter(labels.values()))), "images": len(files), "shape": list(shape),
                    "size_mb": round(files_out["vectors"].stat().st_size / 2**20, 1), "seconds": round(seconds, 1),
                    "images_per_second": round(len(files) / seconds, 1), "created": info["created"]}
                print(f"{dataset}_{feature_set}: {len(files)} images, {report['sets'][f'{dataset}_{feature_set}']['size_mb']} MB, "
                      f"{len(files) / seconds:.0f} images/s")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {report_path}")
    return report

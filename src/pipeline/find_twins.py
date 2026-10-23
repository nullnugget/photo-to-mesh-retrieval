"""The twin step: find near-identical meshes of a dataset by geometry (`src/data/mesh_twins.py`).

Stages (each one reads the output of the one before):

1. grids:     one occupancy grid per gallery entry of the catalog (cells³, dilated), packed bits.
2. pairs:     the grid IoU of every pair of entries (one matrix product on the GPU); the pairs at or
              above `min_iou` go to a table, with their categories.
3. distances: for each pair at or above `min_iou`, first a fast upper bound (nearest dense sample
              points); a pair whose bound is below `exact_below` also gets the exact surface
              distance (about 0.3 seconds per pair on the GPU). A bound at or above `exact_below`
              proves that the pair is not a twin, so the slow test is not needed. Each result is
              written at once, so a stopped run continues where it stopped.

The step finds and measures; it does not merge. The threshold of the merge is chosen from the
measured pairs by a visual review, and the merge itself changes the catalog (a later step).

Outputs (`data/derived/twins/<dataset>/`):
    grids.npz         mesh ids, cells, packed grids
    grid_pairs.csv    mesh_a, mesh_b, category_a, category_b, iou       (iou ≥ the pairs floor)
    distances.csv     mesh_a, mesh_b, category_a, category_b, iou, bound, distance (empty if not measured)
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import paths
from src.data import mesh_twins
from src.data.catalog import read_table

PAIR_COLUMNS = ["mesh_a", "mesh_b", "category_a", "category_b", "iou"]
DISTANCE_COLUMNS = PAIR_COLUMNS + ["bound", "distance"]
EXACT_BELOW = 0.03  # pairs with a bound at or above this are not twins (twins: exact distance about 0.000 to 0.002)


def output_dir(dataset: str) -> Path:
    """Return the folder of the twin files of a dataset."""
    return paths.DERIVED_DIR / "twins" / dataset


def _entries(dataset: str) -> list[dict]:
    """The gallery entries of a dataset (catalog rows), in catalog order."""
    return [row for row in read_table(paths.MESHES_CSV) if row["dataset"] == dataset]


def build_grids(dataset: str, cells: int = mesh_twins.GRID_CELLS, points: int = mesh_twins.GRID_POINTS) -> Path:
    """Stage 1: the occupancy grid of every gallery entry; return the grid file."""
    rows = _entries(dataset)
    grids = np.zeros((len(rows), cells, cells, cells), dtype=bool)
    for i, row in enumerate(tqdm(rows, desc=f"grids {dataset}")):
        mesh = mesh_twins.unit_mesh(paths.resolve(row["mesh_path"]))
        grids[i] = mesh_twins.occupancy_grid(mesh_twins.surface_points(mesh, points, seed=i), cells)
    path = mesh_twins.save_grids(output_dir(dataset) / "grids.npz", [r["mesh_id"] for r in rows], grids)
    print(f"{len(rows)} grids ({cells}³, {points} points each) -> {path.relative_to(paths.ROOT)}", flush=True)
    return path


def grid_pairs(dataset: str, min_iou: float, device: str = "cuda") -> Path:
    """Stage 2: the IoU of all pairs; write the pairs at or above `min_iou`; print the counts by IoU."""
    mesh_ids, grids = mesh_twins.load_grids(output_dir(dataset) / "grids.npz")
    category = {r["mesh_id"]: r["category"] for r in _entries(dataset)}
    iou = mesh_twins.grid_iou(grids, device=device)
    a, b = np.where(np.triu(iou >= min_iou, k=1))
    path = output_dir(dataset) / "grid_pairs.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PAIR_COLUMNS)
        for i, j in sorted(zip(a, b), key=lambda p: -iou[p]):
            writer.writerow([mesh_ids[i], mesh_ids[j], category[mesh_ids[i]], category[mesh_ids[j]], f"{iou[i, j]:.4f}"])
    upper = iou[np.triu_indices(len(iou), k=1)]
    print(f"{len(upper):,} pairs; at or above IoU: " + ", ".join(f"{t} {int((upper >= t).sum()):,}" for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)), flush=True)
    print(f"{len(a):,} pairs (IoU ≥ {min_iou}) -> {path.relative_to(paths.ROOT)}", flush=True)
    return path


def distances(dataset: str, min_iou: float, device: str = "cuda", exact_below: float = EXACT_BELOW) -> Path:
    """Stage 3: the bound, and below `exact_below` the exact surface distance, of the grid pairs at or above
    `min_iou`; continues a stopped run."""
    folder = output_dir(dataset)
    with open(folder / "grid_pairs.csv") as f:
        pairs = [r for r in csv.DictReader(f) if float(r["iou"]) >= min_iou]
    path = folder / "distances.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {(r["mesh_a"], r["mesh_b"]) for r in csv.DictReader(f)}
    todo = [p for p in pairs if (p["mesh_a"], p["mesh_b"]) not in done]
    mesh_path = {r["mesh_id"]: paths.resolve(r["mesh_path"]) for r in _entries(dataset)}
    print(f"{len(pairs):,} pairs with IoU ≥ {min_iou}; {len(done):,} measured before; {len(todo):,} to measure", flush=True)
    cache: dict[str, tuple] = {}

    def mesh(mesh_id):
        """(mesh, measured points, dense points) of one entry; the seeds come from the mesh id, so they repeat."""
        if mesh_id not in cache:
            if len(cache) > 500:  # pairs are sorted by IoU, not by mesh: keep the memory bounded
                cache.clear()
            m = mesh_twins.unit_mesh(mesh_path[mesh_id], device)
            seed = sum(mesh_id.encode())
            cache[mesh_id] = (m, mesh_twins.surface_points(m, mesh_twins.DISTANCE_POINTS, seed),
                              mesh_twins.surface_points(m, mesh_twins.BOUND_POINTS, seed + 1))
        return cache[mesh_id]

    new = not path.exists()
    start = time.perf_counter()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(DISTANCE_COLUMNS)
        for k, p in enumerate(tqdm(todo, desc="distances", mininterval=30)):
            (ma, pa, da), (mb, pb, db) = mesh(p["mesh_a"]), mesh(p["mesh_b"])
            bound = mesh_twins.distance_bound(pa, pb, da, db)
            exact = ""
            if bound < exact_below:
                squared = 0.5 * (float(mesh_twins.point_to_surface(pa, mb).mean()) + float(mesh_twins.point_to_surface(pb, ma).mean()))
                exact = f"{squared ** 0.5:.5f}"
            writer.writerow([p[c] for c in PAIR_COLUMNS] + [f"{bound:.5f}", exact])
            if k % 50 == 0:
                f.flush()
    print(f"{len(todo):,} pairs measured in {(time.perf_counter() - start) / 60:.1f} min -> {path.relative_to(paths.ROOT)}", flush=True)
    return path

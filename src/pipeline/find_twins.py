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

4. merge:    the twin groups at `data.<dataset>.twins.max_distance` (pairs with a grid IoU at or above
              `min_grid_iou`), with every pair of a group within it ("complete link",
              `mesh_twins.complete_link_groups`), become one gallery entry each. A group whose entries
              have different classes takes the class of `data.<dataset>.twins.group_classes` (key: any
              entry of the group: an ABO product id; for Pix3D "<category>/<model folder>", because
              the folder names repeat across categories).
              The representative is the entry of the group's class with the smallest last id part (an
              entry that is rendered already). Written: `twin_groups.csv`, and for ABO also the catalog:
              - ABO: the entries are the groups of exact copies (`catalog_abo_meshes`); the
                stage always starts from their table `geometry_groups_exact.csv`, so it can run again.
                It writes the group table `geometry_groups.csv` (every ABO mesh → its final entry) and
                the ABO rows of `meshes.csv`.
              - Pix3D: the entries are the model folders of pix3d.json; the query step reads
                `twin_groups.csv` and gives the photos and the mesh rows their merged entries.
              Then the query step (`03_make_queries.py --dataset <dataset> --keep-parts`) writes the
              new catalog rows and splits.

Stages 1 to 3 work on the entries before any merge (ABO: the exact copies; Pix3D: the model folders),
so they give the same pairs after a merge. The threshold of stage 4 was chosen from the measured pairs
by a visual review.

Outputs (`data/derived/twins/<dataset>/`):
    grids.npz         mesh ids, cells, packed grids
    grid_pairs.csv    mesh_a, mesh_b, category_a, category_b, iou       (iou ≥ the pairs floor)
    distances.csv     mesh_a, mesh_b, category_a, category_b, iou, bound, distance (empty if not measured)
    twin_groups.csv   group, entry, category, representative, group_category   (stage merge)
Outputs of stage 4 in the catalog (ABO only): `data/raw/abo/manifest/geometry_groups.csv`, the ABO rows
of `data/derived/catalog/meshes.csv`.
"""

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import paths
from src.data import abo, catalog, mesh_twins, pix3d
from src.data.catalog import read_table
from src.pipeline import catalog_abo_meshes, download_abo

PAIR_COLUMNS = ["mesh_a", "mesh_b", "category_a", "category_b", "iou"]
DISTANCE_COLUMNS = PAIR_COLUMNS + ["bound", "distance"]
GROUP_COLUMNS = ["group", "entry", "category", "representative", "group_category"]
TWIN_GROUPS_FILE = "twin_groups.csv"
EXACT_BELOW = 0.03  # pairs with a bound at or above this are not twins (twins: exact distance about 0.000 to 0.002)


def output_dir(dataset: str) -> Path:
    """Return the folder of the twin files of a dataset."""
    return paths.DERIVED_DIR / "twins" / dataset


def exact_entries() -> list[dict]:
    """The ABO entries of the exact-copy groups, as catalog rows, sorted by mesh id.

    Built from `geometry_groups_exact.csv` and the download manifest (not from the catalog, which holds
    the merged entries after stage 4).
    """
    with open(paths.ABO_MANIFEST_DIR / catalog_abo_meshes.EXACT_GROUPS_FILE, newline="") as f:
        groups = list(csv.DictReader(f))
    meshes = {m.mesh_id: m for m in abo.read_manifest(paths.ABO_MANIFEST_DIR)[0]}
    size = {}
    for row in groups:
        size[row["group_mesh_id"]] = int(row["group_size"])
    return [{"mesh_id": entry, "dataset": abo.DATASET, "category": meshes[entry].category,
             "mesh_path": str(download_abo.mesh_file(paths.ABO_MESHES_DIR, meshes[entry]).relative_to(paths.ROOT)),
             "render_dir": str((paths.RENDERS_DIR / entry).relative_to(paths.ROOT)),
             "num_photos": 0, "num_mesh_variants": size[entry]} for entry in sorted(size)]


def _entries(dataset: str) -> list[dict]:
    """The gallery entries that the stages compare, before any merge: for ABO the exact-copy entries,
    for Pix3D the model folders of pix3d.json (not the catalog, which holds the merged entries)."""
    if dataset == abo.DATASET:
        return exact_entries()
    if dataset == pix3d.DATASET:
        return [{"mesh_id": m.mesh_id, "dataset": dataset, "category": m.category, "mesh_path": m.mesh_path,
                 "num_mesh_variants": m.num_mesh_variants} for m in pix3d.mesh_records(pix3d.load_annotations())]
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


def measured_distances(dataset: str, min_iou: float = 0.0) -> dict[frozenset, float]:
    """The distance of every measured pair with a grid IoU at or above `min_iou`: the exact value, or the
    bound if the exact test was skipped (the bound is never smaller than the exact value, so a skipped
    pair stays "far" for any threshold below the bound)."""
    with open(output_dir(dataset) / "distances.csv") as f:
        return {frozenset((r["mesh_a"], r["mesh_b"])): float(r["distance"] or r["bound"])
                for r in csv.DictReader(f) if float(r["iou"]) >= min_iou}


def read_twin_groups(dataset: str) -> dict[str, tuple[str, str]]:
    """Entry → (representative, class of its group) for every entry of a twin group of the merge stage;
    empty if the dataset has no merge."""
    path = output_dir(dataset) / TWIN_GROUPS_FILE
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["entry"]: (r["representative"], r["group_category"]) for r in csv.DictReader(f)}


def merge(twins_cfg: dict, dataset: str = abo.DATASET) -> dict:
    """Stage 4: join the twin groups into one gallery entry each; write `twin_groups.csv`, and for ABO
    also the group table and the ABO catalog rows (Pix3D: the query step applies `twin_groups.csv`).

    Args:
        twins_cfg: `data.<dataset>.twins` (max_distance, min_grid_iou, group_classes).
        dataset: "abo" or "pix3d".

    Returns:
        A report: entries before and after, groups, meshes in groups, the classes chosen by the settings.

    Raises:
        ValueError: If a group with several classes has no entry in `group_classes`, if a key of
            `group_classes` is in no group or two keys are in one group, or if the chosen class has no
            entry in the group.
    """
    if dataset not in (abo.DATASET, pix3d.DATASET):
        raise ValueError(f"no merge for dataset {dataset!r}")
    folder = output_dir(dataset)
    with open(folder / "grid_pairs.csv") as f:
        min_iou = min(float(r["iou"]) for r in csv.DictReader(f))
    if min_iou > twins_cfg["min_grid_iou"] + 1e-9:
        raise ValueError(f"grid_pairs.csv starts at IoU {min_iou}, above min_grid_iou {twins_cfg['min_grid_iou']}")
    entries = {row["mesh_id"]: row for row in _entries(dataset)}
    if dataset == abo.DATASET:
        with open(paths.ABO_MANIFEST_DIR / catalog_abo_meshes.EXACT_GROUPS_FILE, newline="") as f:
            exact = list(csv.DictReader(f))
        entry_of_key = {row["mesh_id"].split("/")[-1]: row["group_mesh_id"] for row in exact}
    else:
        entry_of_key = {e.split("/", 1)[1]: e for e in entries}  # "<category>/<model folder>": folders repeat across categories

    groups = mesh_twins.complete_link_groups(measured_distances(dataset, twins_cfg["min_grid_iou"]), twins_cfg["max_distance"])
    chosen = dict(twins_cfg.get("group_classes") or {})
    used, final_of, out_rows = set(), {}, []
    for k, group in enumerate(groups):
        classes = {entries[e]["category"] for e in group}
        keys = [p for p in chosen if entry_of_key.get(p) in group]
        if len(keys) > 1:
            raise ValueError(f"group {sorted(group)}: two keys in group_classes {keys}")
        if len(classes) > 1 and not keys:
            raise ValueError(f"group {sorted(group)} has the classes {sorted(classes)}: add the key of one entry to data.{dataset}.twins.group_classes")
        category = chosen[keys[0]] if keys else classes.pop()
        used.update(keys)
        candidates = sorted(e for e in group if entries[e]["category"] == category)
        if not candidates:
            raise ValueError(f"group {sorted(group)}: no entry of the chosen class {category!r}")
        lead = min(candidates, key=lambda e: e.split("/")[-1])
        for e in sorted(group):
            final_of[e] = lead
            out_rows.append({"group": k, "entry": e, "category": entries[e]["category"], "representative": lead,
                             "group_category": category})
    unused = sorted(set(chosen) - used)
    if unused:
        raise ValueError(f"data.{dataset}.twins.group_classes: keys in no group {unused}")
    with open(folder / TWIN_GROUPS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GROUP_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)
    after = len(entries) - len(final_of) + len(groups)
    report = {"entries_before": len(entries), "entries_after": after, "groups": len(groups),
              "entries_in_groups": sum(len(g) for g in groups), "group_classes": chosen,
              "max_distance": twins_cfg["max_distance"], "min_grid_iou": twins_cfg["min_grid_iou"]}
    if dataset == abo.DATASET:
        _write_abo_catalog(exact, entries, final_of)
        written = f"{TWIN_GROUPS_FILE}, {catalog_abo_meshes.GROUPS_FILE}, meshes.csv"
    else:
        written = f"{TWIN_GROUPS_FILE} (next: 03_make_queries.py --dataset {dataset} --keep-parts)"
    print(f"merge: {len(entries)} → {after} {dataset} entries ({len(groups)} twin groups of {report['entries_in_groups']} entries; "
          f"max distance {twins_cfg['max_distance']}, complete link) -> {written}", flush=True)
    return report


def _write_abo_catalog(exact: list[dict], entries: dict[str, dict], final_of: dict[str, str]) -> None:
    """The ABO group table (every ABO mesh → its final entry) and the ABO catalog rows (one per final entry)."""
    members = defaultdict(list)
    for row in exact:
        members[final_of.get(row["group_mesh_id"], row["group_mesh_id"])].append(row)
    group_rows, mesh_rows = [], []
    for lead, rows in sorted(members.items()):
        for row in rows:
            group_rows.append({"mesh_id": row["mesh_id"], "group_mesh_id": lead, "category": row["category"],
                               "group_category": entries[lead]["category"], "group_size": len(rows)})
        mesh_rows.append({**entries[lead], "num_mesh_variants": len(rows)})
    group_rows.sort(key=lambda r: r["mesh_id"])
    with open(paths.ABO_MANIFEST_DIR / catalog_abo_meshes.GROUPS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=catalog_abo_meshes.GROUP_COLUMNS)
        writer.writeheader()
        writer.writerows(group_rows)
    catalog.replace_dataset_rows(paths.MESHES_CSV, mesh_rows, abo.DATASET, catalog.MESH_COLUMNS)

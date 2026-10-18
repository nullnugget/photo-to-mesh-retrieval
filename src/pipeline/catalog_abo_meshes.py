"""Put the downloaded ABO meshes into the catalog, with each distinct geometry only once.

About a quarter of the ABO meshes have the same geometry as another mesh: colour or material
variants of one product, sold under their own product ids. With the grey material, identical
geometry gives identical renders, so a search could not tell such meshes apart, and the "wrong"
copy would count as an error by chance. Thus each distinct geometry becomes one gallery entry:

1. Candidates: two meshes can be copies only if they have the same number of vertices, the same
   number of triangles, and exactly the same triangle list (the same vertex numbers in the same
   order). A hash of the triangle list makes this comparison fast.
2. Same shape: every mesh is scaled to the unit sphere (as for rendering; so a copy that is only
   scaled up or down still counts as a copy). Two candidates are copies if no vertex differs by
   more than 0.002 of the sphere radius (about half a pixel in a render). The comparison uses 2,048
   evenly spaced vertices of each mesh, which keeps the memory small.
   Why not an exact hash of the positions: real copies differ by tiny rounding errors (up to about
   0.001), so an exact hash splits them. And meshes with equal counts but different positions (for
   example a left and a right version of a corner sofa) are different products and stay apart.
3. Group: copies form one group. The group's catalog entry uses the mesh of the group's most common
   class with the smallest product id ("representative").
4. Catalog: one row per group in `data/derived/catalog/meshes.csv` (the ABO rows are replaced).
   `num_mesh_variants` is the number of meshes in the group. `num_photos` is 0 until the ABO query
   step adds the photos.
5. Group table: `data/raw/abo/manifest/geometry_groups.csv` maps every ABO mesh to its group's
   entry, so that the query step can give each photo the right gallery entry.

The ABO meshes can be rendered as soon as their catalog rows exist; the photos (which need masks)
are added later by the query step.
"""

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import paths
from src.data import abo, catalog
from src.render.mesh_io import load_mesh
from src.render.normalize import normalize_to_unit_sphere

SAME_SHAPE_TOLERANCE = 0.002  # largest vertex difference of copies, in unit-sphere units
SAMPLED_VERTICES = 2048  # vertices compared per mesh
GROUP_COLUMNS = ["mesh_id", "group_mesh_id", "category", "group_category", "group_size"]


def geometry_signature(mesh_path: str | Path) -> tuple[tuple, np.ndarray]:
    """Return (candidate key, vertex sample) of one mesh (see steps 1 and 2 of the module description).

    The key is (vertex count, triangle count, hash of the triangle list). The sample holds
    SAMPLED_VERTICES evenly spaced vertices of the mesh, scaled to the unit sphere: float32 (n, 3).
    """
    vertices, faces = load_mesh(mesh_path)
    normalized = normalize_to_unit_sphere(vertices)[0].numpy()
    key = (len(vertices), len(faces), hashlib.sha1(faces.numpy().astype(np.int32).tobytes()).hexdigest())
    sample = normalized[np.linspace(0, len(normalized) - 1, SAMPLED_VERTICES).astype(int)]
    return key, sample.astype(np.float32)


def read_geometry_groups(manifest_dir: Path) -> dict[str, str]:
    """Read `geometry_groups.csv` (written by `run`): {mesh id: mesh id of its group's gallery entry}."""
    with open(manifest_dir / "geometry_groups.csv", newline="") as f:
        return {row["mesh_id"]: row["group_mesh_id"] for row in csv.DictReader(f)}


def photo_entry(photo: abo.AboImage, groups: dict[str, str]) -> str | None:
    """Return the gallery entry (group mesh id) that a photo shows, or None if there is no single one.

    A photo belongs to the listings of one or more meshes. If all of them are in the same geometry
    group (copies of one geometry), the photo shows that group's entry. If they are in different
    groups (for example a photo of a whole product line), the photo has no single correct answer.
    A mesh without a group (its file was missing) also gives None.
    """
    entries = {groups.get(mesh_id) for mesh_id in photo.mesh_ids}
    return next(iter(entries)) if len(entries) == 1 and None not in entries else None


def _representative(members: list[abo.AboMesh]) -> abo.AboMesh:
    """The mesh that represents a group: from the group's most common class, the smallest product id."""
    top_class = Counter(m.category for m in members).most_common(1)[0][0]
    return min((m for m in members if m.category == top_class), key=lambda m: m.model_id)


def run(meshes: list[abo.AboMesh], mesh_folder: Path, mesh_file, catalog_dir: Path, manifest_dir: Path) -> dict:
    """Fingerprint and group the meshes, and write the catalog rows and the group table.

    Args:
        meshes: The selected ABO meshes (from the manifest).
        mesh_folder: Folder of the PLY files.
        mesh_file: Function (folder, mesh) -> PLY path (`download_abo.mesh_file`).
        catalog_dir: Folder of `meshes.csv` (the real catalog, or a test folder for a mini run).
        manifest_dir: Folder of `geometry_groups.csv`.

    Returns:
        A report: meshes, missing files, groups, merged meshes, groups that span classes.
    """
    # Steps 1 to 3: within each set of candidates, a mesh joins the first group whose first mesh has
    # the same shape, or starts a new group.
    candidates: dict[tuple, list[tuple[np.ndarray, list[abo.AboMesh]]]] = defaultdict(list)
    missing = []
    for mesh in tqdm(meshes, desc="compare geometry", unit="mesh"):
        path = mesh_file(mesh_folder, mesh)
        if not path.exists():
            missing.append(mesh.mesh_id)
            continue
        key, sample = geometry_signature(path)
        for first_sample, members in candidates[key]:
            if np.abs(sample - first_sample).max() <= SAME_SHAPE_TOLERANCE:
                members.append(mesh)
                break
        else:  # no group with the same shape
            candidates[key].append((sample, [mesh]))
    groups = [members for groups_of_key in candidates.values() for _, members in groups_of_key]

    rows, group_rows, mixed_classes = [], [], []
    for members in groups:
        lead = _representative(members)
        if len({m.category for m in members}) > 1:
            mixed_classes.append(sorted(m.mesh_id for m in members))
        rows.append({
            "mesh_id": lead.mesh_id,
            "dataset": abo.DATASET,
            "category": lead.category,
            "mesh_path": str(mesh_file(mesh_folder, lead).relative_to(paths.ROOT)),
            "render_dir": str((paths.RENDERS_DIR / lead.mesh_id).relative_to(paths.ROOT)),
            "num_photos": 0,  # the ABO query step fills this in
            "num_mesh_variants": len(members),
        })
        for member in members:
            group_rows.append({"mesh_id": member.mesh_id, "group_mesh_id": lead.mesh_id, "category": member.category,
                               "group_category": lead.category, "group_size": len(members)})

    rows.sort(key=lambda r: r["mesh_id"])
    group_rows.sort(key=lambda r: r["mesh_id"])
    catalog.replace_dataset_rows(catalog_dir / paths.MESHES_CSV.name, rows, abo.DATASET, catalog.MESH_COLUMNS)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "geometry_groups.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GROUP_COLUMNS)
        writer.writeheader()
        writer.writerows(group_rows)

    report = {
        "meshes": len(meshes),
        "missing_files": missing,
        "gallery_entries": len(rows),
        "merged_meshes": len(meshes) - len(missing) - len(rows),  # meshes that joined another mesh's entry
        "groups_with_copies": sum(len(m) > 1 for m in groups),
        "groups_spanning_classes": mixed_classes,
        "entries_per_class": dict(Counter(r["category"] for r in rows).most_common()),
    }
    print(f"catalog: {len(rows)} ABO gallery entries from {len(meshes) - len(missing)} meshes "
          f"({report['merged_meshes']} merged into {report['groups_with_copies']} groups; "
          f"{len(mixed_classes)} groups span classes; {len(missing)} files missing)")
    return report

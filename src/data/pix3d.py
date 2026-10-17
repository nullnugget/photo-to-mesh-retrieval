"""Reader for the Pix3D dataset.

Pix3D (Sun et al., CVPR 2018) pairs real photos of furniture with the exact 3D mesh of the
object in each photo. Its annotation file, `data/raw/pix3d/pix3d.json`, is a list with one
entry for each photo. The entries used here:

- "img":      path of the photo, for example "img/chair/0001.png"
- "mask":     path of the object mask (white = object), for example "mask/chair/0001.png"
- "model":    path of the 3D mesh, for example "model/chair/IKEA_POANG_2/model.obj"
- "category": object class, for example "chair"
- "truncated", "occluded", "slightly_occluded": quality flags of the photo, set by hand by the
  Pix3D authors

All paths in the file are relative to `data/raw/pix3d/`.

Identifiers used by the project:

- A photo is identified as "pix3d/<category>/<file name without extension>", for example
  "pix3d/chair/0001". This is unique: no two photos in a category share a file name.
- A mesh is identified as "pix3d/<category>/<model folder>", for example
  "pix3d/chair/IKEA_POANG_2". The category is part of the identifier because some folder
  names appear in two categories (for example IKEA_EKTORP_1 is a chair and a sofa).

Two photos have broken masks: the mask file belongs to another photo (other size, other
object or pose). They are "img/table/0045.png" and "img/table/1749.jpg". The query step
(`src/preprocess/queries.py`) finds them by the size difference and leaves them out.

Five chair folders (for example IKEA_MARKUS) contain one posed mesh file for each photo
(a swivel chair at different angles). They are still one mesh entry, with several pose
variants. The reader uses the first variant in annotation order as the mesh file of the
entry, and counts the variants.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from src import paths

DATASET = "pix3d"


@dataclass(frozen=True)
class PhotoRecord:
    """Everything the project needs to know about one Pix3D photo.

    `photo_records` makes one record for each entry of pix3d.json. The query step uses the
    records to make the query images, and writes each record as one row of the query table of
    the catalog (`data/derived/catalog/queries.csv`). Paths are relative to the repository root.
    """

    query_id: str  # identifier of the photo, "pix3d/<category>/<name>"
    category: str  # object class, for example "chair"
    mesh_id: str  # identifier of the mesh in the photo, "pix3d/<category>/<model folder>"
    photo_path: str  # the original photo
    mask_path: str  # the object mask of the photo (white = object)
    truncated: bool  # True if the photo border cuts the object
    occluded: bool  # True if other objects hide part of the object
    slightly_occluded: bool  # True if other objects hide a small part of the object


@dataclass(frozen=True)
class MeshRecord:
    """Everything the project needs to know about one Pix3D mesh entry.

    `mesh_records` makes one record for each distinct model folder in pix3d.json. The query
    step writes each record as one row of the mesh table of the catalog
    (`data/derived/catalog/meshes.csv`). The render step reads that table to know which mesh
    files to render. Paths are relative to the repository root.
    """

    mesh_id: str  # identifier, "pix3d/<category>/<model folder>"
    category: str  # object class, for example "chair"
    folder: str  # model folder name, for example "IKEA_POANG_2"
    mesh_path: str  # the mesh file to render (the first variant for the swivel chairs)
    num_photos: int  # number of photos of this mesh in pix3d.json
    num_mesh_variants: int  # number of mesh files: 1 for most entries; more for the swivel chairs


def load_annotations(json_path: Path = paths.PIX3D_JSON) -> list[dict]:
    """Read pix3d.json and return its list of photo entries, in file order."""
    with open(json_path) as f:
        return json.load(f)


def _relative_to_root(pix3d_relative: str) -> str:
    """Change a path relative to data/raw/pix3d/ into a path relative to the repository root."""
    return str((paths.PIX3D_DIR / pix3d_relative).relative_to(paths.ROOT))


def query_id_of(entry: dict) -> str:
    """Return the photo identifier, "pix3d/<category>/<name>", of one annotation entry."""
    return f"{DATASET}/{entry['category']}/{Path(entry['img']).stem}"


def mesh_id_of(entry: dict) -> str:
    """Return the mesh identifier, "pix3d/<category>/<model folder>", of one annotation entry."""
    folder = Path(entry["model"]).parent.name  # "model/chair/IKEA_POANG_2/model.obj" -> "IKEA_POANG_2"
    return f"{DATASET}/{entry['category']}/{folder}"


def photo_records(annotations: list[dict]) -> list[PhotoRecord]:
    """Make one PhotoRecord for each annotation entry, in file order."""
    return [
        PhotoRecord(
            query_id=query_id_of(entry),
            category=entry["category"],
            mesh_id=mesh_id_of(entry),
            photo_path=_relative_to_root(entry["img"]),
            mask_path=_relative_to_root(entry["mask"]),
            truncated=bool(entry.get("truncated", False)),
            occluded=bool(entry.get("occluded", False)),
            slightly_occluded=bool(entry.get("slightly_occluded", False)),
        )
        for entry in annotations
    ]


def mesh_records(annotations: list[dict]) -> list[MeshRecord]:
    """Make one MeshRecord for each distinct mesh entry, in order of first appearance."""
    first_mesh: dict[str, str] = {}  # mesh id -> first mesh file seen
    variants: dict[str, set[str]] = {}  # mesh id -> all distinct mesh files
    photo_count: dict[str, int] = {}  # mesh id -> number of photos
    category: dict[str, str] = {}
    for entry in annotations:
        mid = mesh_id_of(entry)
        first_mesh.setdefault(mid, entry["model"])  # setdefault keeps the first value only
        variants.setdefault(mid, set()).add(entry["model"])
        photo_count[mid] = photo_count.get(mid, 0) + 1
        category[mid] = entry["category"]
    return [
        MeshRecord(
            mesh_id=mid,
            category=category[mid],
            folder=mid.split("/")[-1],
            mesh_path=_relative_to_root(first_mesh[mid]),
            num_photos=photo_count[mid],
            num_mesh_variants=len(variants[mid]),
        )
        for mid in first_mesh
    ]


def v1_test_query_ids(annotations: list[dict], seed: int = 42, holdout_fraction: float = 0.15) -> list[str]:
    """Return the photos that the v1 code held out for its evaluation, as query ids.

    The v1 code shuffled the entries of pix3d.json and kept the last 15 % apart:
    `random.seed(42); random.shuffle(entries); cut = int(len(entries) * 0.85)`.
    `random.Random(42).shuffle` gives the same order as that code. The order of the entries in
    pix3d.json therefore matters, so this function takes the annotations in file order.

    The seen-meshes split of Pix3D uses these photos as its test part, so that v1 and v2 results
    are measured on the same photos. The list has 1,511 photos. One of them, "pix3d/table/0045",
    has a broken mask and thus no query image; the split leaves it out. That gives the 1,510
    photos that v1 evaluated.
    """
    shuffled = [query_id_of(entry) for entry in annotations]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - holdout_fraction))
    return shuffled[cut:]

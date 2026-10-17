"""Reader for the Pix3D dataset.

Pix3D (Sun et al., CVPR 2018) pairs real photos of furniture with the exact 3D shape of the
object in each photo. Its annotation file, `data/raw/pix3d/pix3d.json`, is a list with one
entry for each photo. The entries used here:

- "img":      path of the photo, for example "img/chair/0001.png"
- "mask":     path of the object mask (white = object), for example "mask/chair/0001.png"
- "model":    path of the 3D mesh, for example "model/chair/IKEA_POANG_2/model.obj"
- "category": object class, for example "chair"
- "truncated", "occluded", "slightly_occluded": quality flags of the photo

All paths in the file are relative to `data/raw/pix3d/`.

Identifiers used by the project:

- A photo is identified as "pix3d/<category>/<file name without extension>", for example
  "pix3d/chair/0001". This is unique: no two photos in a category share a file name.
- A shape is identified as "pix3d/<category>/<model folder>", for example
  "pix3d/chair/IKEA_POANG_2". The category is part of the identifier because some folder
  names appear in two categories (for example IKEA_EKTORP_1 is a chair and a sofa).

Five chair folders (for example IKEA_MARKUS) contain one posed mesh variant for each photo
(a swivel chair at different angles). They are still one shape. The reader uses the first
variant in annotation order as the mesh of the shape, and counts the variants.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src import paths

DATASET = "pix3d"


@dataclass(frozen=True)
class PhotoRecord:
    """One Pix3D photo and its annotations. Paths are relative to the repository root."""

    query_id: str  # "pix3d/<category>/<name>"
    category: str
    shape_id: str  # "pix3d/<category>/<model folder>"
    photo_path: str
    mask_path: str
    truncated: bool
    occluded: bool
    slightly_occluded: bool


@dataclass(frozen=True)
class ShapeRecord:
    """One Pix3D 3D shape. Paths are relative to the repository root."""

    shape_id: str  # "pix3d/<category>/<model folder>"
    category: str
    folder: str  # model folder name, for example "IKEA_POANG_2"
    mesh_path: str  # the mesh that represents the shape (the first variant for swivel chairs)
    num_photos: int  # number of photos of this shape
    num_mesh_variants: int  # 1 for most shapes; more for the swivel chairs


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


def shape_id_of(entry: dict) -> str:
    """Return the shape identifier, "pix3d/<category>/<model folder>", of one annotation entry."""
    folder = Path(entry["model"]).parent.name  # "model/chair/IKEA_POANG_2/model.obj" -> "IKEA_POANG_2"
    return f"{DATASET}/{entry['category']}/{folder}"


def photo_records(annotations: list[dict]) -> list[PhotoRecord]:
    """Make one PhotoRecord for each annotation entry, in file order."""
    return [
        PhotoRecord(
            query_id=query_id_of(entry),
            category=entry["category"],
            shape_id=shape_id_of(entry),
            photo_path=_relative_to_root(entry["img"]),
            mask_path=_relative_to_root(entry["mask"]),
            truncated=bool(entry.get("truncated", False)),
            occluded=bool(entry.get("occluded", False)),
            slightly_occluded=bool(entry.get("slightly_occluded", False)),
        )
        for entry in annotations
    ]


def shape_records(annotations: list[dict]) -> list[ShapeRecord]:
    """Make one ShapeRecord for each distinct shape, in order of first appearance."""
    first_mesh: dict[str, str] = {}  # shape id -> first mesh path seen
    variants: dict[str, set[str]] = {}  # shape id -> all distinct mesh paths
    photo_count: dict[str, int] = {}  # shape id -> number of photos
    category: dict[str, str] = {}
    for entry in annotations:
        sid = shape_id_of(entry)
        first_mesh.setdefault(sid, entry["model"])  # setdefault keeps the first value only
        variants.setdefault(sid, set()).add(entry["model"])
        photo_count[sid] = photo_count.get(sid, 0) + 1
        category[sid] = entry["category"]
    return [
        ShapeRecord(
            shape_id=sid,
            category=category[sid],
            folder=sid.split("/")[-1],
            mesh_path=_relative_to_root(first_mesh[sid]),
            num_photos=photo_count[sid],
            num_mesh_variants=len(variants[sid]),
        )
        for sid in first_mesh
    ]


def read_photo_and_mask(photo_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one photo and its mask, ready for the crop.

    Args:
        photo_path: Photo file, relative to the repository root (or absolute).
        mask_path: Mask file, relative to the repository root (or absolute).

    Returns:
        image: uint8 array of shape (H, W, 3), RGB.
        mask: bool array of shape (H, W); True = object.

    Photos come in several formats and modes (JPEG, PNG, one TIFF; grayscale, RGBA, palette).
    Pillow reads all of them, and `convert("RGB")` gives 3 channels in every case.
    A few masks do not have the size of their photo. Such a mask is resized to the photo size
    with nearest-neighbour sampling, which keeps it strictly black and white.
    """
    with Image.open(paths.resolve(photo_path)) as photo:
        image = photo.convert("RGB")
    with Image.open(paths.resolve(mask_path)) as mask_file:
        mask = mask_file.convert("L")  # one grey channel, values 0..255
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    # Mask pixels above the middle grey are object pixels.
    return np.asarray(image, dtype=np.uint8), np.asarray(mask) > 127


if __name__ == "__main__":
    # Stand-alone check (from the repository root: python -m src.data.pix3d).
    # It reads the annotations, counts photos and shapes, and reads the photo whose mask size differs.
    from collections import Counter

    annotations = load_annotations()
    photos = photo_records(annotations)
    shapes = shape_records(annotations)
    print(f"photos: {len(photos)} | shapes: {len(shapes)}")
    print("photos per category:", dict(Counter(p.category for p in photos)))
    print("shapes with several mesh variants:", [(s.shape_id, s.num_mesh_variants) for s in shapes if s.num_mesh_variants > 1])
    example = next(p for p in photos if p.query_id == "pix3d/table/0045")  # its mask is smaller than the photo
    image, mask = read_photo_and_mask(example.photo_path, example.mask_path)
    print(f"{example.query_id}: image {image.shape}, mask {mask.shape}, object pixels {mask.mean():.1%}")

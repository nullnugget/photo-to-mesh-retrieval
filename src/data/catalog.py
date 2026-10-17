"""The catalog: one table of queries and one table of meshes, for all datasets.

Every dataset reader (Pix3D now, ABO later) writes its rows into the same two tables. All later
stages (rendering, feature extraction, training, evaluation) read only these tables. Thus a later
stage does not need to know which dataset an item comes from, and a new dataset needs only its own
reader and preparation script.

Files (CSV, one row per item, paths relative to the repository root):

- `data/derived/catalog/queries.csv`, one row per query photo:
    query_id           "pix3d/chair/0001"
    dataset            "pix3d"
    category           "chair"
    mesh_id            the mesh of the object in the photo, "pix3d/chair/IKEA_POANG_2"
    photo_path         the original photo
    mask_path          the object mask of the photo
    query_path         the prepared query image (masked, cropped, 224 x 224)
    truncated          True if the object is cut by the photo border
    occluded           True if other objects hide part of the object
    slightly_occluded  True if other objects hide a small part of the object

- `data/derived/catalog/meshes.csv`, one row per 3D mesh entry (one object of the gallery):
    mesh_id            "pix3d/chair/IKEA_POANG_2"
    dataset            "pix3d"
    category           "chair"
    mesh_path          the mesh file to render
    render_dir         the folder of its renders (view_00.png ... view_24.png)
    num_photos         number of query photos of this mesh
    num_mesh_variants  number of mesh files of this entry (more than 1 only for posed variants)

The three photo flags come from the dataset annotations; the project does not compute them.
Pix3D has them (set by hand by its authors). A dataset without such annotations (ABO) leaves
the cells empty, and `read_table` returns None for them: "unknown", not "False". The flags are
used only to break down the results (for example the recall on occluded photos), not in training.
"""

import csv
from pathlib import Path
from typing import Iterable

QUERY_COLUMNS = [
    "query_id",
    "dataset",
    "category",
    "mesh_id",
    "photo_path",
    "mask_path",
    "query_path",
    "truncated",
    "occluded",
    "slightly_occluded",
]

MESH_COLUMNS = [
    "mesh_id",
    "dataset",
    "category",
    "mesh_path",
    "render_dir",
    "num_photos",
    "num_mesh_variants",
]

# Columns that are converted back from text when a table is read.
_BOOL_COLUMNS = {"truncated", "occluded", "slightly_occluded"}
_INT_COLUMNS = {"num_photos", "num_mesh_variants"}


def write_table(rows: Iterable[dict], path: Path, columns: list[str]) -> Path:
    """Write rows to a CSV file with the given columns, in this order.

    Each row must have exactly these columns: a missing or an extra column is an error, so that
    a mistake cannot write an incomplete catalog. A value of None is written as an empty cell.
    """
    rows = list(rows)
    for row in rows:
        if set(row) != set(columns):
            raise KeyError(f"row columns {sorted(row)} differ from the table columns {sorted(columns)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_table(path: Path) -> list[dict]:
    """Read a catalog table. Boolean and integer columns get their Python types back.

    An empty cell in a boolean column (a flag that the dataset does not annotate) becomes None.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for column in _BOOL_COLUMNS & set(row):
            row[column] = None if row[column] == "" else row[column] == "True"
        for column in _INT_COLUMNS & set(row):
            row[column] = int(row[column])
    return rows


def replace_dataset_rows(path: Path, rows: Iterable[dict], dataset: str, columns: list[str]) -> Path:
    """Replace all rows of one dataset in a table, and keep the rows of the other datasets.

    This lets each dataset preparation script update its own part of the shared catalog.
    If the table does not exist yet, it is created with only the new rows.
    """
    rows = list(rows)
    if any(row["dataset"] != dataset for row in rows):
        raise ValueError(f"all new rows must belong to dataset {dataset!r}")
    kept = [row for row in read_table(path) if row["dataset"] != dataset] if Path(path).exists() else []
    return write_table(kept + rows, path, columns)

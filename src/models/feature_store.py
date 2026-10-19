"""The feature store: backbone features of renders and query images, calculated once and read many times.

The backbone is frozen, so the same image always gives the same features. The extraction step
(`scripts/05_extract_features.py`) calculates them once on the GPU and saves them here; the
evaluation and the training then read the files in seconds, without the backbone.

One file holds one feature set of one dataset for one backbone
(`data/derived/features/<backbone>/<dataset>_<set>.npz`, see `paths.features_file`):

- renders: one row per gallery mesh; each row has the vectors of all its views.
    cls, masked_mean   float16 (meshes, views, dim)
    mesh_id, category  text (meshes,)
- queries (queries_reference, queries_sam): one row per query image, for every photo of the dataset.
  A split file selects the rows of its parts by query_id.
    cls, masked_mean           float16 (queries, dim)
    query_id, mesh_id, category  text (queries,)

Every file also holds `info`: a small JSON record of how the file was made (backbone, weights, image
source, settings, date, git commit), so that a copied file still tells where it comes from.

Why both basic parts ("cls" and "masked_mean", see `src/models/backbones.py`): one pass through the
backbone gives both, and every feature recipe is made from them (`FeatureTable.vectors`). So one
extraction serves the comparison of the recipes.

Why float16: the vectors have length 1, so float16 keeps about 3 correct digits of each number,
which does not change the rankings in practice, at half the size of float32.

Why `.npz`: one file keeps several arrays together (the vectors and their IDs). The file is not
compressed, because float16 vectors compress very little. Text is stored as fixed-width unicode
arrays, so the files load without pickle (pickle can run code from a file).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.backbones import FEATURE_PARTS, combine_recipe

RENDERS = "renders"
QUERIES = "queries"
# Labels that every row must have: one row per mesh for renders, one row per query image for queries.
LABELS = {RENDERS: ("mesh_id", "category"), QUERIES: ("query_id", "mesh_id", "category")}
# The label that names each row; it must be unique in the file.
ROW_ID = {RENDERS: "mesh_id", QUERIES: "query_id"}
STORE_DTYPE = np.float16


@dataclass
class FeatureTable:
    """The content of one feature file.

    Attributes:
        kind: RENDERS or QUERIES.
        parts: "cls" and "masked_mean" -> float16 vectors; (meshes, views, dim) or (queries, dim).
        labels: The text labels of the rows (see LABELS) -> arrays of length rows.
        info: How the file was made (free-form JSON-compatible record).
    """

    kind: str
    parts: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    info: dict

    @property
    def rows(self) -> int:
        """Number of rows: meshes for renders, query images for queries."""
        return len(self.labels[ROW_ID[self.kind]])

    @property
    def ids(self) -> np.ndarray:
        """The row IDs: mesh_id for renders, query_id for queries."""
        return self.labels[ROW_ID[self.kind]]

    def vectors(self, recipe: str) -> np.ndarray:
        """Return the feature vectors of one recipe, float32 and unit length (see `combine_recipe`)."""
        return combine_recipe(self.parts, recipe)

    def select(self, ids) -> "FeatureTable":
        """Return a table with only the rows of `ids`, in the order of `ids` (for example one split part).

        Raises:
            KeyError: If an ID is not in the table.
        """
        row_of = {row_id: index for index, row_id in enumerate(self.ids)}
        missing = [row_id for row_id in ids if row_id not in row_of]
        if missing:
            raise KeyError(f"{len(missing)} IDs are not in the table, for example {missing[:3]}")
        index = np.array([row_of[row_id] for row_id in ids], dtype=np.int64)
        return FeatureTable(
            self.kind,
            {name: values[index] for name, values in self.parts.items()},
            {name: values[index] for name, values in self.labels.items()},
            dict(self.info),
        )


def check_table(table: FeatureTable) -> None:
    """Stop with an error if the table is not complete and consistent.

    Raises:
        ValueError: An unknown kind, missing or extra parts or labels, arrays of different lengths,
            a wrong number of axes, or a repeated row ID.
    """
    if table.kind not in LABELS:
        raise ValueError(f"kind must be one of {tuple(LABELS)}, got {table.kind!r}")
    if set(table.parts) != set(FEATURE_PARTS):
        raise ValueError(f"parts must be {FEATURE_PARTS}, got {sorted(table.parts)}")
    if set(table.labels) != set(LABELS[table.kind]):
        raise ValueError(f"labels must be {LABELS[table.kind]}, got {sorted(table.labels)}")
    axes = 3 if table.kind == RENDERS else 2  # renders: (meshes, views, dim); queries: (queries, dim)
    shapes = {name: values.shape for name, values in table.parts.items()}
    if len(set(shapes.values())) != 1 or len(next(iter(shapes.values()))) != axes:
        raise ValueError(f"the parts must have the same shape with {axes} axes, got {shapes}")
    rows = next(iter(shapes.values()))[0]
    lengths = {name: len(values) for name, values in table.labels.items()}
    if any(length != rows for length in lengths.values()):
        raise ValueError(f"every label needs {rows} rows, got {lengths}")
    if len(set(table.ids.tolist())) != rows:
        raise ValueError(f"the {ROW_ID[table.kind]} values are not unique")


def write_features(table: FeatureTable, path: Path) -> Path:
    """Check the table and save it as one `.npz` file; return the path.

    The vectors are saved as float16. The file is written under a temporary name and then renamed,
    so that an interrupted run never leaves a partial file under the real name.
    """
    check_table(table)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"part_{name}": np.asarray(values, dtype=STORE_DTYPE) for name, values in table.parts.items()}
    arrays.update({f"label_{name}": np.asarray(values, dtype=str) for name, values in table.labels.items()})
    arrays["kind"] = np.asarray(table.kind)
    arrays["info"] = np.asarray(json.dumps(table.info, sort_keys=True))
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as f:  # a file object: np.savez would add ".npz" to a file name
        np.savez(f, **arrays)
    os.replace(partial, path)  # an atomic rename on the same disk
    return path


def read_features(path: Path) -> FeatureTable:
    """Read one feature file (as written by `write_features`) and check it."""
    with np.load(path, allow_pickle=False) as data:  # allow_pickle=False: the file can hold no code
        table = FeatureTable(
            kind=str(data["kind"]),
            parts={key[len("part_"):]: data[key] for key in data.files if key.startswith("part_")},
            labels={key[len("label_"):]: data[key] for key in data.files if key.startswith("label_")},
            info=json.loads(str(data["info"])),
        )
    check_table(table)
    return table

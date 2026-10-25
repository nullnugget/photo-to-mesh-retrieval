"""The patch store: the patch vectors of every render and query image, for the re-ranking (stage 2).

The feature store (`feature_store.py`) keeps one vector per image (the class token). A re-ranker
compares a photo and a mesh part by part, so it needs the 14 x 14 = 196 patch tokens of each image
(one vector per 16 x 16 square). At 768 numbers each, all images would need about 37 GB. The store
reduces each patch vector with PCA (principal component analysis) to `rerank.patches.dim` numbers:

    stored = (patch - mean) @ components[:, :dim]

- mean, components: fitted once on patch vectors of non-test renders and photos (`fit_projection`
  in `src/pipeline/extract_patches.py`); one fixed projection for photos and renders.
- The components are sorted by the variance they explain, so the first 64 of 128 stored numbers are
  exactly the 64-number reduction. A user of the store can take fewer numbers without a new pass.
- Before a comparison, each stored vector is scaled to length 1 again (cosine similarity).

Files of one set (`data/derived/patches/<backbone>/`, see `patch_files`):

| File                         | Content                                                         |
|------------------------------|-----------------------------------------------------------------|
| `<dataset>_<set>.npy`        | float16 vectors: (meshes, views, 196, dim) or (queries, 196, dim) |
| `<dataset>_<set>_object.npy` | uint8 (same leading axes, 196): share of object pixels x 255     |
| `<dataset>_<set>_labels.npz` | the row labels (as in the feature file, same order) and `info`    |
| `projection.npz`             | mean, components (all 768), explained variance                  |

The rows have the same order as the feature file of the same set, so row i of the patch store and
row i of the feature store are the same image.

Why `.npy` and not `.npz`: a `.npy` file can be opened as a memory map (`np.load(mmap_mode="r")`),
so a reader takes only the rows it needs (for example the 10 candidates of one photo) and the
6 GB store never has to fit in memory.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src import paths
from src.models.feature_store import LABELS, QUERIES, RENDERS, ROW_ID

PATCH_DTYPE = np.float16
PROJECTION_FILE = "projection.npz"


def patch_files(backbone: str, dataset: str, feature_set: str, root: Path = paths.PATCHES_DIR) -> dict[str, Path]:
    """Return the three files of one set: {"vectors", "object", "labels"}."""
    folder = root / backbone
    return {"vectors": folder / f"{dataset}_{feature_set}.npy",
            "object": folder / f"{dataset}_{feature_set}_object.npy",
            "labels": folder / f"{dataset}_{feature_set}_labels.npz"}


def projection_file(backbone: str, root: Path = paths.PATCHES_DIR) -> Path:
    return root / backbone / PROJECTION_FILE


# ---------------------------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------------------------


class CovarianceSum:
    """Sums of patch vectors for a PCA, added batch by batch (the patch vectors never have to be kept).

    The covariance is E[x xᵀ] − E[x] E[x]ᵀ; both terms come from running sums in float64.
    """

    def __init__(self, dim: int, device: str | torch.device = "cpu"):
        self.count = 0
        self.total = torch.zeros(dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def add(self, vectors: torch.Tensor) -> None:
        """Add vectors (n, dim)."""
        x = vectors.to(self.total.device, torch.float64)
        self.count += len(x)
        self.total += x.sum(0)
        self.outer += x.T @ x

    def fit(self) -> "PatchProjection":
        """Return the PCA of the added vectors: all components, sorted by explained variance (largest first)."""
        if self.count < 2:
            raise ValueError("the PCA needs at least 2 vectors")
        mean = self.total / self.count
        covariance = self.outer / self.count - torch.outer(mean, mean)
        variance, components = torch.linalg.eigh(covariance)  # ascending order
        order = torch.argsort(variance, descending=True)
        return PatchProjection(mean.float().cpu().numpy(), components[:, order].float().cpu().numpy(),
                               variance[order].clamp(min=0).float().cpu().numpy(), self.count)


@dataclass
class PatchProjection:
    """A fitted PCA: mean (dim,), components (dim, dim) as columns, variance (dim,), and the fit count."""

    mean: np.ndarray
    components: np.ndarray
    variance: np.ndarray
    count: int

    def explained(self, dim: int) -> float:
        """Share of the total variance that the first `dim` components keep."""
        return float(self.variance[:dim].sum() / self.variance.sum())

    def project(self, vectors: torch.Tensor, dim: int) -> torch.Tensor:
        """(..., full dim) -> (..., dim): (x - mean) @ components[:, :dim], on the device of `vectors`."""
        mean = torch.as_tensor(self.mean, device=vectors.device, dtype=torch.float32)
        weights = torch.as_tensor(self.components[:, :dim], device=vectors.device, dtype=torch.float32)
        return (vectors.float() - mean) @ weights

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components, variance=self.variance, count=self.count)
        return path

    @classmethod
    def load(cls, path: Path) -> "PatchProjection":
        with np.load(path, allow_pickle=False) as data:
            return cls(data["mean"], data["components"], data["variance"], int(data["count"]))


# ---------------------------------------------------------------------------------------------
# Store files
# ---------------------------------------------------------------------------------------------


class PatchWriter:
    """Writes one set batch by batch into memory-mapped files under temporary names.

    `close` renames the files to their real names, so an interrupted run never leaves a partial
    file under the real name.
    """

    def __init__(self, files: dict[str, Path], shape: tuple[int, ...], dim: int, patches: int):
        self.files = files
        self.partial = {key: path.with_name(path.name + ".part") for key, path in files.items()}
        files["vectors"].parent.mkdir(parents=True, exist_ok=True)
        self.vectors = np.lib.format.open_memmap(self.partial["vectors"], mode="w+", dtype=PATCH_DTYPE,
                                                 shape=(*shape, patches, dim))
        self.object = np.lib.format.open_memmap(self.partial["object"], mode="w+", dtype=np.uint8,
                                                shape=(*shape, patches))
        self.flat_vectors = self.vectors.reshape(-1, patches, dim)  # image by image, in file order
        self.flat_object = self.object.reshape(-1, patches)
        self.written = 0

    def add(self, vectors: np.ndarray, object_share: np.ndarray) -> None:
        """Add the next images: vectors (n, patches, dim), object share in [0, 1] (n, patches)."""
        n = len(vectors)
        self.flat_vectors[self.written:self.written + n] = vectors.astype(PATCH_DTYPE)
        self.flat_object[self.written:self.written + n] = np.round(object_share * 255).astype(np.uint8)
        self.written += n

    def close(self, kind: str, labels: dict[str, np.ndarray], info: dict) -> None:
        if self.written != len(self.flat_vectors):
            raise ValueError(f"{self.written} of {len(self.flat_vectors)} images written")
        self.vectors.flush(); self.object.flush()
        del self.vectors, self.object, self.flat_vectors, self.flat_object
        with open(self.partial["labels"], "wb") as f:  # a file object: np.savez would add ".npz" to the name
            np.savez(f, kind=np.asarray(kind), info=np.asarray(json.dumps(info, sort_keys=True)),
                     **{f"label_{name}": np.asarray(values, dtype=str) for name, values in labels.items()})
        for key, path in self.files.items():
            os.replace(self.partial[key], path)


@dataclass
class PatchTable:
    """One set of the store. `vectors` and `object` are read-only memory maps (rows load on use)."""

    kind: str
    vectors: np.ndarray  # float16 (meshes, views, 196, dim) or (queries, 196, dim)
    object: np.ndarray  # uint8 share x 255, same leading axes, (…, 196)
    labels: dict[str, np.ndarray]
    info: dict

    @property
    def ids(self) -> np.ndarray:
        return self.labels[ROW_ID[self.kind]]

    def row_index(self) -> dict[str, int]:
        return {row_id: i for i, row_id in enumerate(self.ids)}


def read_patches(files: dict[str, Path]) -> PatchTable:
    """Open one set of the store (memory maps; nothing large is read here)."""
    with np.load(files["labels"], allow_pickle=False) as data:
        kind = str(data["kind"])
        labels = {key[len("label_"):]: data[key] for key in data.files if key.startswith("label_")}
        info = json.loads(str(data["info"]))
    if kind not in (RENDERS, QUERIES) or set(labels) != set(LABELS[kind]):
        raise ValueError(f"{files['labels']}: kind {kind!r} with labels {sorted(labels)}")
    table = PatchTable(kind, np.load(files["vectors"], mmap_mode="r"), np.load(files["object"], mmap_mode="r"), labels, info)
    if len(table.vectors) != len(table.ids) or table.object.shape != table.vectors.shape[:-1]:
        raise ValueError(f"{files['vectors']}: shapes {table.vectors.shape} / {table.object.shape} do not match {len(table.ids)} rows")
    return table

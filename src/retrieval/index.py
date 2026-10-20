"""The gallery index of a trained checkpoint: the embedded gallery, saved once, and the search scores.

After training, the heads do not change. So the view embeddings and the mesh embeddings of every
gallery mesh are calculated once and saved; a search then embeds only the query photo.

One index file per checkpoint file and dataset, next to the checkpoint:

    weights/checkpoints/<run_name>/embedding_heads_<best|last>_gallery_<dataset>.npz
        mesh_ids     (M,)          the gallery meshes, in the order of the renders feature file
        categories   (M,)
        views        (M, V, D)     float32: the trained view embeddings (view projector)
        meshes       (M, D)        float32: the trained mesh embeddings (view aggregator)
        info         JSON text: the checkpoint (path, SHA-256), the renders feature file (path,
                     SHA-256), the backbone, the feature recipe, and the scoring settings

The final score mixes the trained score with the zero-shot score of the raw features:

    final = mix_weight × trained score + (1 − mix_weight) × zero-shot score
    trained score   = alpha × best view (q · e_v) + (1 − alpha) × mesh (q · g)   query embedding q
    zero-shot score = the zero-shot scoring of `backbone.zero_shot` on the raw features

The zero-shot part needs the raw view features of the gallery. They are not copied into the index:
the index names the renders feature file and its SHA-256, and loading the index reads that file
and stops if it has changed. Both parts come from one backbone pass of the query photo: the raw
feature is the zero-shot query vector, and the query projector turns it into q.

Why an exact search and no FAISS: every mesh gets its exact score, so the index gives the same
R@1 as the validation during training (the check of `scripts/07_build_index.py`). The view score
takes the maximum over the views of a mesh in two different spaces (trained and raw), so a FAISS
search for the top views of one space can miss the mesh that the mixed score puts first. At our
size the exact search is fast: Pix3D has 9,875 views; Pix3D and ABO together about 130,000.
An approximate index is useful only for millions of views.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src import paths, runs
from src.models.embedding_heads import EmbeddingHeads, scores
from src.models.feature_store import FeatureTable, read_features
from src.retrieval.evaluate import score_meshes


class IndexMismatch(RuntimeError):
    """The renders feature file of an index has changed since the index was built."""


@dataclass
class GalleryIndex:
    """The embedded gallery of one dataset, and the raw view features for the zero-shot part."""

    mesh_ids: np.ndarray  # (M,)
    categories: np.ndarray  # (M,)
    views: np.ndarray  # float32 (M, V, D): trained view embeddings
    meshes: np.ndarray  # float32 (M, D): trained mesh embeddings
    raw_views: np.ndarray  # float32 (M, V, F): raw backbone features of the views (the chosen recipe)
    info: dict


def index_file(checkpoint: Path, dataset: str) -> Path:
    """Return the index file of a checkpoint file for one dataset."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(f"{checkpoint.stem}_gallery_{dataset}.npz")


def load_heads(checkpoint: dict, device: str = "cpu") -> EmbeddingHeads:
    """Return the heads of a loaded checkpoint (`runs.load_checkpoint`), in evaluation mode."""
    feature_dim = checkpoint["query_projector"]["layers.0.weight"].shape[1]  # the input length of the first layer
    heads = EmbeddingHeads.from_config(checkpoint["meta"]["config"]["train"]["heads"], feature_dim)
    for key in runs.HEAD_KEYS:
        getattr(heads, key).load_state_dict(checkpoint[key])
    return heads.to(device).eval()


def _gallery_vectors(renders: FeatureTable, recipe: str) -> np.ndarray:
    """The raw view features in float16 precision, as float32: the training keeps the gallery in
    float16 (`src/train/trainer.py`), so the index gives the same scores as the validation."""
    return renders.vectors(recipe).astype(np.float16).astype(np.float32)


@torch.no_grad()
def build_index(heads: EmbeddingHeads, renders: FeatureTable, recipe: str, info: dict, device: str = "cpu") -> GalleryIndex:
    """Embed every view and every mesh of a renders feature table with the heads.

    The stored features go to the heads as float16, as in the training, so that the embeddings are
    the ones that the validation used.
    """
    raw_views = _gallery_vectors(renders, recipe)
    views, meshes = heads.embed_meshes(torch.from_numpy(raw_views).to(device, torch.float16))
    return GalleryIndex(
        mesh_ids=renders.ids, categories=renders.labels["category"],
        views=views.float().cpu().numpy(), meshes=meshes.float().cpu().numpy(),
        raw_views=raw_views, info=info,
    )


def save_index(index: GalleryIndex, path: Path) -> Path:
    """Write the index (without the raw view features; see the module description)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part.npz")
    np.savez(partial, mesh_ids=index.mesh_ids.astype(str), categories=index.categories.astype(str),
             views=index.views, meshes=index.meshes, info=np.array(json.dumps(index.info)))
    partial.replace(path)  # a stopped write leaves no half file under the real name
    return path


def load_index(path: Path) -> GalleryIndex:
    """Read an index and the raw view features of its renders feature file.

    Raises:
        IndexMismatch: If the renders feature file has changed (another SHA-256 or other meshes).
    """
    with np.load(path, allow_pickle=False) as data:
        stored = {key: data[key] for key in data.files}
    info = json.loads(str(stored["info"]))
    renders_file = paths.resolve(info["renders_file"])
    if runs.file_sha256(renders_file) != info["renders_sha256"]:
        raise IndexMismatch(f"{renders_file} has changed since the index {path} was built; build the index again")
    renders = read_features(renders_file)
    if renders.ids.tolist() != stored["mesh_ids"].tolist():
        raise IndexMismatch(f"the meshes of {renders_file} differ from the index {path}")
    return GalleryIndex(stored["mesh_ids"], stored["categories"], stored["views"], stored["meshes"],
                        _gallery_vectors(renders, info["recipe"]), info)


@torch.no_grad()
def search_scores(index: GalleryIndex, heads: EmbeddingHeads, raw_queries: np.ndarray, alpha: float,
                  zero_shot: dict, device: str = "cpu") -> dict[str, np.ndarray]:
    """Return the trained and the zero-shot score of every gallery mesh for every query: float32 (Q, M) each.

    Args:
        raw_queries: (Q, F), the raw backbone features of the query photos (unit length, the recipe of the index).
        alpha: Weight of the view score in the trained score.
        zero_shot: {"scoring", "alpha"} of the zero-shot score (`backbone.zero_shot`).

    `mix_scores` combines the two into the final score.
    """
    q = heads.embed_queries(torch.from_numpy(raw_queries).to(device, torch.float16))
    trained = scores(q, torch.from_numpy(index.views).to(device), torch.from_numpy(index.meshes).to(device), alpha)
    return {"trained": trained.float().cpu().numpy(),
            "zero_shot": score_meshes(raw_queries, index.raw_views, zero_shot["scoring"], zero_shot["alpha"])}


def mix_scores(search: dict[str, np.ndarray], mix_weight: float) -> np.ndarray:
    """Final score: mix_weight × trained + (1 − mix_weight) × zero-shot (1.0 = trained alone, 0.0 = zero-shot)."""
    return (mix_weight * search["trained"] + (1.0 - mix_weight) * search["zero_shot"]).astype(np.float32)

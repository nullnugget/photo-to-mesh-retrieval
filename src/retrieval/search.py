"""Scores between query features and the views of each mesh.

Each mesh has one feature vector per view (25 views). All vectors have length 1, so a dot product
is a cosine similarity. Three scores compare a query with a mesh:

- view score: the similarity to the best-matching view of the mesh (the maximum over its views).
  A photo shows the object from one direction, so one view should match it well.
- mesh score: the similarity to the mean of the mesh's view vectors (scaled to length 1). It
  summarizes the mesh from all directions; it is more stable but less sharp.
- combined score: alpha x view score + (1 - alpha) x mesh score.

Before training, the scores use the raw backbone features ("zero-shot"). After training, the same
functions take the outputs of the embedding heads. With a few thousand meshes, the exact scores
fit in memory, so no approximate index (FAISS) is needed for the evaluation.
"""

import numpy as np


def view_scores(queries: np.ndarray, views: np.ndarray) -> np.ndarray:
    """Return the view score of every query against every mesh: (Q, M).

    Args:
        queries: float (Q, D), unit length.
        views: float (M, V, D): V unit-length view vectors for each of M meshes.
    """
    similarities = np.einsum("qd,mvd->qmv", queries, views)  # every query against every view of every mesh
    return similarities.max(axis=2)


def mesh_vectors(views: np.ndarray) -> np.ndarray:
    """Return one unit-length vector per mesh, the normalized mean of its views: (M, D)."""
    mean = views.mean(axis=1)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True).clip(min=1e-12)


def mesh_scores(queries: np.ndarray, views: np.ndarray) -> np.ndarray:
    """Return the mesh score of every query against every mesh: (Q, M)."""
    return queries @ mesh_vectors(views).T


def combined_scores(queries: np.ndarray, views: np.ndarray, alpha: float) -> np.ndarray:
    """Return alpha x view score + (1 - alpha) x mesh score: (Q, M)."""
    return alpha * view_scores(queries, views) + (1.0 - alpha) * mesh_scores(queries, views)

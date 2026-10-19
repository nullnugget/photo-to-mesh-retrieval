"""Evaluation: rank every gallery mesh for each query photo, and compute the retrieval metrics.

The input is one vector per query and one vector per view of every gallery mesh. In zero-shot mode
these are the raw backbone features of the feature store; after training they are the outputs of
the embedding heads. The functions here do not care which, as long as all vectors have length 1.

The gallery is always every mesh of the dataset, also the meshes that no photo shows (a photo must
find its mesh among all meshes, as in real use).

Steps for one setting (recipe and scoring):

1. score every mesh for every query (`src/retrieval/search.py`: view, mesh, or combined score);
2. find the rank of the correct mesh, overall and among the meshes of the query's category;
3. compute the metrics (`src/retrieval/metrics.py`): R@1, R@5, R@10 with the 95 % interval of R@1,
   Cat@1, R@1 inside the category, per-category R@1, macro R@1, and R@1 of the photo subsets
   (clean, truncated, occluded; only for datasets that annotate them);
4. keep one row per query (ranks and first result), for the failure analysis.
"""

from dataclasses import dataclass

import numpy as np

from src.retrieval import metrics, search

SCORINGS = ("view", "mesh", "combined")
TOP_K = (1, 5, 10)
# Photo subsets from the dataset's own annotations: name -> rule on the flags of one photo.
SUBSETS = {
    "clean": lambda f: not f["truncated"] and not f["occluded"] and not f["slightly_occluded"],
    "truncated": lambda f: bool(f["truncated"]),
    "occluded": lambda f: bool(f["occluded"]),
    "slightly_occluded": lambda f: bool(f["slightly_occluded"]),
}


@dataclass
class Queries:
    """The query side of an evaluation: one unit-length vector and three labels per query."""

    vectors: np.ndarray  # float (Q, D)
    query_ids: np.ndarray  # (Q,)
    mesh_ids: np.ndarray  # (Q,): the correct mesh of each query
    categories: np.ndarray  # (Q,)


@dataclass
class Gallery:
    """The mesh side of an evaluation: the unit-length vectors of the views of every mesh."""

    views: np.ndarray  # float (M, V, D)
    mesh_ids: np.ndarray  # (M,)
    categories: np.ndarray  # (M,)


def score_meshes(query_vectors: np.ndarray, views: np.ndarray, scoring: str, alpha: float | None = None) -> np.ndarray:
    """Return the score of every mesh for every query: float32 (Q, M).

    Args:
        query_vectors: (Q, D), unit length.
        views: (M, V, D), unit length.
        scoring: "view" (best view), "mesh" (mean of the views), or "combined".
        alpha: Weight of the view score in the combined score; needed only for "combined".
    """
    if scoring == "view":
        return search.view_scores(query_vectors, views)
    if scoring == "mesh":
        return search.mesh_scores(query_vectors, views)
    if scoring == "combined":
        if alpha is None:
            raise ValueError("the combined score needs alpha")
        return search.combined_scores(query_vectors, views, alpha)
    raise ValueError(f"scoring must be one of {SCORINGS}, got {scoring!r}")


def _subset_recall(ranks: np.ndarray, query_ids: np.ndarray, flags: dict[str, dict] | None) -> dict[str, dict]:
    """R@1 of each photo subset. Empty if the photos have no annotations (flags missing or None)."""
    if not flags or any(flags[q]["truncated"] is None for q in query_ids):
        return {}
    result = {}
    for name, rule in SUBSETS.items():
        member = np.array([rule(flags[q]) for q in query_ids])
        if member.any():
            result[name] = {"queries": int(member.sum()), "R@1": metrics.recall_at_k(ranks[member], 1)}
    return result


def evaluate(queries: Queries, gallery: Gallery, scoring: str = "view", alpha: float | None = None,
             flags: dict[str, dict] | None = None) -> tuple[dict, list[dict]]:
    """Evaluate one setting; return (metrics, one row per query).

    Args:
        queries: The query vectors and labels.
        gallery: The view vectors of every gallery mesh.
        scoring: One of SCORINGS.
        alpha: Weight of the view score (only for "combined").
        flags: query_id -> {"truncated", "occluded", "slightly_occluded"} (True, False, or None = not
            annotated), for the subset results. None = no subsets.

    Raises:
        KeyError: If the correct mesh of a query is not in the gallery.
    """
    column = {mesh_id: index for index, mesh_id in enumerate(gallery.mesh_ids.tolist())}
    missing = sorted({m for m in queries.mesh_ids.tolist() if m not in column})
    if missing:
        raise KeyError(f"{len(missing)} correct meshes are not in the gallery, for example {missing[:3]}")
    correct = np.array([column[m] for m in queries.mesh_ids.tolist()])

    scores = score_meshes(queries.vectors, gallery.views, scoring, alpha)
    ranks = metrics.correct_ranks(scores, correct)
    ranks_inside = metrics.ranks_inside_category(scores, correct, queries.categories, gallery.categories)
    category_hits = metrics.category_hits(scores, queries.categories, gallery.categories)
    low, high = metrics.bootstrap_interval(ranks <= 1)

    result = {f"R@{k}": metrics.recall_at_k(ranks, k) for k in TOP_K}
    result.update({
        "photos": len(ranks),
        "gallery_meshes": len(gallery.mesh_ids),
        "R@1_low": low,  # 95 % bootstrap interval of R@1
        "R@1_high": high,
        "cat@1": float(category_hits.mean()),
        "R@1_inside_category": metrics.recall_at_k(ranks_inside, 1),
        "macro_R@1": metrics.macro_recall(ranks, queries.categories),
        "per_category": metrics.recall_by_group(ranks, queries.categories),
        "subsets": _subset_recall(ranks, queries.query_ids, flags),
    })

    first = scores.argmax(axis=1)  # the column of the first result of every query
    rows = [
        {
            "query_id": str(queries.query_ids[i]),
            "category": str(queries.categories[i]),
            "mesh_id": str(queries.mesh_ids[i]),
            "rank": int(ranks[i]),
            "rank_inside_category": int(ranks_inside[i]),
            "first_mesh_id": str(gallery.mesh_ids[first[i]]),
            "first_score": round(float(scores[i, first[i]]), 4),
            "correct_score": round(float(scores[i, correct[i]]), 4),
        }
        for i in range(len(ranks))
    ]
    return result, rows

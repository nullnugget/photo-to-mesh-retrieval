"""Retrieval metrics and their confidence intervals.

For each query, the meshes are sorted by score, highest first. The rank of the correct mesh is
its position in that list (1 = first).

- R@K (recall at K): the share of queries whose correct mesh is among the first K. R@1 is the
  main metric: the system must return exactly the right mesh first.
- Cat@1 (category accuracy at 1): the share of queries whose first result has the right category
  (for example any chair for a chair photo). It shows whether the errors are near misses
  (a similar chair) or gross errors (a table for a chair).
- R@1 inside the category: the R@1 when each query is compared only with the meshes of its own
  category. The difference to R@1 is the share of errors that pick a mesh of a wrong category.
- Per-category R@1 and macro R@1: the R@1 of each category, and their plain mean. A large category
  (Pix3D: chairs) dominates R@1; macro R@1 gives every category the same weight.

Confidence interval (bootstrap): the queries are a sample, so a metric has random error. The
bootstrap draws the same number of queries again at random, with replacement, many times, and
computes the metric each time. The middle 95 % of these values is the 95 % interval. To compare
two settings on the same queries, the paired bootstrap draws the same queries for both, so that
the shared difficulty of the queries cancels out.
"""

import numpy as np


def correct_ranks(scores: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Return the rank (1 = best) of the correct mesh for every query: int (Q,).

    Args:
        scores: (Q, M), higher = better match.
        correct: int (Q,), the column of the correct mesh for each query.

    A tie with the correct mesh counts against it (pessimistic), so a constant score cannot look good.
    """
    correct_score = scores[np.arange(len(scores)), correct][:, None]
    return (scores >= correct_score).sum(axis=1)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Share of queries whose correct mesh is among the first k results."""
    return float((ranks <= k).mean())


def category_hits(scores: np.ndarray, query_categories: np.ndarray, mesh_categories: np.ndarray) -> np.ndarray:
    """Return True for every query whose first result has the query's category: bool (Q,)."""
    return mesh_categories[scores.argmax(axis=1)] == query_categories


def ranks_inside_category(scores: np.ndarray, correct: np.ndarray, query_categories: np.ndarray, mesh_categories: np.ndarray) -> np.ndarray:
    """Return the rank of the correct mesh among the meshes of the query's own category: int (Q,).

    The meshes of other categories get the score minus infinity, so they rank after every mesh of
    the right category and cannot take the correct mesh's place.
    """
    same_category = mesh_categories[None, :] == query_categories[:, None]  # (Q, M)
    return correct_ranks(np.where(same_category, scores, -np.inf), correct)


def recall_by_group(ranks: np.ndarray, groups: np.ndarray, k: int = 1) -> dict[str, dict]:
    """Return R@k and the number of queries for each group (for example each category).

    Returns:
        {group: {"queries": n, "R@k": share}}, sorted by group name.
    """
    return {
        str(group): {"queries": int((groups == group).sum()), f"R@{k}": recall_at_k(ranks[groups == group], k)}
        for group in sorted(set(groups.tolist()))
    }


def macro_recall(ranks: np.ndarray, groups: np.ndarray, k: int = 1) -> float:
    """Return the mean of the per-group R@k values: every group counts the same."""
    return float(np.mean([entry[f"R@{k}"] for entry in recall_by_group(ranks, groups, k).values()]))


def bootstrap_interval(hits: np.ndarray, samples: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """95 % bootstrap interval of the mean of per-query hits (for example `ranks <= 1`)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hits), size=(samples, len(hits)))  # each row: one resample of the queries
    means = hits[draws].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def paired_difference(hits_a: np.ndarray, hits_b: np.ndarray, samples: int = 2000, seed: int = 0) -> dict:
    """Compare two settings on the same queries: mean(b) - mean(a), with its 95 % interval.

    Returns:
        {"difference", "low", "high", "share_b_better"}: share_b_better is the share of resamples in
        which b scores higher than a (near 1.0 or near 0.0 = a clear difference).
    """
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hits_a), size=(samples, len(hits_a)))
    differences = hits_b[draws].astype(float).mean(axis=1) - hits_a[draws].astype(float).mean(axis=1)
    return {
        "difference": float(hits_b.mean() - hits_a.mean()),
        "low": float(np.quantile(differences, 0.025)),
        "high": float(np.quantile(differences, 0.975)),
        "share_b_better": float((differences > 0).mean()),
    }

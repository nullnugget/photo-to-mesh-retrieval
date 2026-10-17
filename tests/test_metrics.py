"""Tests for src/retrieval/search.py and src/retrieval/metrics.py, on hand-made numbers."""

import numpy as np

from src.retrieval.metrics import bootstrap_interval, category_hits, correct_ranks, paired_difference, recall_at_k
from src.retrieval.search import combined_scores, mesh_scores, view_scores


def _unit(*vectors):
    array = np.array(vectors, dtype=float)
    return array / np.linalg.norm(array, axis=-1, keepdims=True)


def test_view_mesh_and_combined_scores():
    query = _unit([1.0, 0.0])[0]
    views = np.stack([_unit([1.0, 0.0], [0.0, 1.0]), _unit([0.8, 0.6], [0.8, 0.6])])  # mesh 0: one perfect view; mesh 1: two good views
    assert np.allclose(view_scores(query[None], views), [[1.0, 0.8]])
    mesh = mesh_scores(query[None], views)
    assert np.allclose(mesh, [[np.sqrt(0.5), 0.8]])  # the mean of mesh 0's views points at 45 degrees
    assert np.allclose(combined_scores(query[None], views, 0.5), (view_scores(query[None], views) + mesh) / 2)


def test_ranks_recall_and_ties():
    scores = np.array([[0.9, 0.5, 0.1], [0.2, 0.7, 0.7], [0.3, 0.3, 0.3]])
    ranks = correct_ranks(scores, np.array([0, 1, 2]))
    assert list(ranks) == [1, 2, 3]  # ties count against the correct mesh
    assert recall_at_k(ranks, 1) == 1 / 3 and recall_at_k(ranks, 3) == 1.0


def test_category_hits():
    scores = np.array([[0.9, 0.1], [0.9, 0.1]])
    hits = category_hits(scores, np.array(["chair", "table"]), np.array(["chair", "table"]))
    assert list(hits) == [True, False]


def test_bootstrap_contains_the_mean_and_paired_difference_sign():
    hits = np.array([1, 0] * 50, dtype=float)
    low, high = bootstrap_interval(hits)
    assert low < 0.5 < high
    better = paired_difference(np.zeros(100), np.r_[np.ones(30), np.zeros(70)])
    assert better["difference"] == 0.3 and better["low"] > 0 and better["share_b_better"] == 1.0

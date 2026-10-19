"""Tests for the scores in src/retrieval/search.py and their dispatch in src/retrieval/evaluate.py.

The main example is small enough to check by hand. Query q = (1, 0). Two meshes, three views each:

    mesh A: (1, 0), (0, 1), (-1, 0)      one perfect view, the others poor
    mesh B: (0.6, 0.8), (0.8, 0.6), (0.6, 0.8)   three good views, none perfect

- View score (best view): A = max(1, 0, -1) = 1.0; B = max(0.6, 0.8, 0.6) = 0.8. A wins.
- Mesh score (mean of the views, scaled to length 1):
  A: mean (0, 1/3) -> (0, 1) -> score 0.0.
  B: mean (2/3, 2.2/3) -> length sqrt(8.84)/3 -> (0.67267, 0.73994) -> score 0.67267. B wins.
- Combined, alpha = 0.5: A = 0.5; B = (0.8 + 0.67267) / 2 = 0.73634. B wins.
"""

import numpy as np
import pytest

from src.retrieval import search
from src.retrieval.evaluate import score_meshes

QUERY = np.array([[1.0, 0.0]])
VIEWS = np.array([
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],  # mesh A
    [[0.6, 0.8], [0.8, 0.6], [0.6, 0.8]],  # mesh B
])
B_MESH_SCORE = 2 / np.sqrt(8.84)  # 0.67267


def test_view_score_is_the_best_view_of_each_mesh():
    assert np.allclose(search.view_scores(QUERY, VIEWS), [[1.0, 0.8]])


def test_mesh_score_uses_the_unit_mean_of_the_views():
    vectors = search.mesh_vectors(VIEWS)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], [0.0, 1.0])
    assert np.allclose(search.mesh_scores(QUERY, VIEWS), [[0.0, B_MESH_SCORE]])


def test_view_and_mesh_scores_can_rank_differently():
    assert search.view_scores(QUERY, VIEWS).argmax() == 0  # A: one perfect view
    assert search.mesh_scores(QUERY, VIEWS).argmax() == 1  # B: good from every side


@pytest.mark.parametrize("alpha, expected", [(1.0, [1.0, 0.8]), (0.0, [0.0, B_MESH_SCORE]), (0.5, [0.5, (0.8 + B_MESH_SCORE) / 2])])
def test_combined_score_mixes_view_and_mesh_scores(alpha, expected):
    assert np.allclose(search.combined_scores(QUERY, VIEWS, alpha), [expected])


def test_scores_have_one_row_per_query_and_one_column_per_mesh():
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(7, 5))
    views = rng.normal(size=(4, 3, 5))
    for scores in (search.view_scores(queries, views), search.mesh_scores(queries, views), search.combined_scores(queries, views, 0.3)):
        assert scores.shape == (7, 4) and scores.dtype == np.float32


def test_view_scores_do_not_depend_on_the_chunk_size():
    rng = np.random.default_rng(1)
    queries, views = rng.normal(size=(10, 6)), rng.normal(size=(5, 4, 6))
    assert np.allclose(search.view_scores(queries, views, chunk=3), search.view_scores(queries, views, chunk=1024))


def test_float16_inputs_give_float32_scores():
    scores = search.view_scores(QUERY.astype(np.float16), VIEWS.astype(np.float16))
    assert scores.dtype == np.float32 and np.allclose(scores, [[1.0, 0.8]], atol=1e-3)


def test_score_meshes_dispatches_and_checks_its_arguments():
    assert np.allclose(score_meshes(QUERY, VIEWS, "view"), search.view_scores(QUERY, VIEWS))
    assert np.allclose(score_meshes(QUERY, VIEWS, "mesh"), search.mesh_scores(QUERY, VIEWS))
    assert np.allclose(score_meshes(QUERY, VIEWS, "combined", 0.5), search.combined_scores(QUERY, VIEWS, 0.5))
    with pytest.raises(ValueError):
        score_meshes(QUERY, VIEWS, "combined")  # alpha missing
    with pytest.raises(ValueError):
        score_meshes(QUERY, VIEWS, "best")

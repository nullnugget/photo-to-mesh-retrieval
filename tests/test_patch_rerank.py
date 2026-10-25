"""Tests for src/retrieval/patch_rerank.py (re-ranking method 1) and the view choice of the re-ranking step."""

import numpy as np
import torch

from src.pipeline.rerank import best_views
from src.retrieval.patch_rerank import patch_scores, reranked_ranks


def _one_hot(*indices, dim=4):
    return torch.eye(dim)[list(indices)]


def test_a_view_with_the_same_patches_scores_1():
    photo = _one_hot(0, 1, 2)[None]  # 1 photo, 3 patches
    views = _one_hot(2, 0, 1)[None, None, None]  # the same patches in another order
    s = patch_scores(photo, torch.ones(1, 3, dtype=torch.bool), views, torch.ones(1, 1, 1, 3, dtype=torch.bool))
    assert s.shape == (1, 1, 1) and torch.allclose(s, torch.ones(1, 1, 1))


def test_both_directions_count():
    photo = _one_hot(0, 1)[None]  # leg, seat
    extra = _one_hot(0, 1, 2, 3)[None, None, None]  # leg, seat and two parts the photo does not show
    s = patch_scores(photo, torch.ones(1, 2, dtype=torch.bool), extra, torch.ones(1, 1, 1, 4, dtype=torch.bool))
    # photo -> view: both found (1.0); view -> photo: 2 of 4 found, 2 match nothing (0.0): mean 0.5
    assert torch.allclose(s, torch.tensor(0.75))


def test_background_patches_do_not_count():
    photo = _one_hot(0, 3)[None]  # the second patch is background
    views = _one_hot(0, 2)[None, None, None]  # the second patch is background
    s = patch_scores(photo, torch.tensor([[True, False]]), views, torch.tensor([[[[True, False]]]]))
    assert torch.allclose(s, torch.ones(1, 1, 1))


def test_an_image_without_object_patches_uses_all_patches():
    photo = _one_hot(0, 1)[None]
    views = _one_hot(0, 1)[None, None, None]
    s = patch_scores(photo, torch.zeros(1, 2, dtype=torch.bool), views, torch.zeros(1, 1, 1, 2, dtype=torch.bool))
    assert torch.allclose(s, torch.ones(1, 1, 1))


def test_reranking_moves_the_correct_mesh_and_keeps_photos_outside_the_list():
    top_index = np.array([[5, 7, 9], [1, 2, 3], [4, 6, 8]])
    truth = np.array([7, 1, 0])  # 2nd, 1st, not in the list
    stage1_rank = np.array([2, 1, 40])
    top_score = np.array([[0.80, 0.78, 0.70], [0.9, 0.5, 0.4], [0.9, 0.8, 0.7]])
    patch = np.array([[0.50, 0.60, 0.90], [0.5, 0.6, 0.7], [0.0, 0.0, 0.0]])
    assert list(reranked_ranks(stage1_rank, top_index, truth, top_score, patch, 0.0, 3)) == [2, 1, 40]
    # weight 1: photo 1: 1.30 / 1.38 / 1.60 -> the correct mesh (1.38) is 2nd; photo 2: 1.4 / 1.1 / 1.1 -> stays 1st
    assert list(reranked_ranks(stage1_rank, top_index, truth, top_score, patch, 1.0, 3)) == [2, 1, 40]
    # weight 0.2: 0.90 / 0.90 / 0.88: a tie counts against the correct mesh -> 2
    assert list(reranked_ranks(stage1_rank, top_index, truth, top_score, patch, 0.2, 3)) == [2, 1, 40]
    # weight 0.5: 1.05 / 1.08 / 1.15 -> 2nd
    patch2 = patch.copy(); patch2[0] = [0.1, 0.9, 0.2]  # 0.85 / 1.23 / 0.80 at weight 0.5 -> 1st
    assert list(reranked_ranks(stage1_rank, top_index, truth, top_score, patch2, 0.5, 3)) == [1, 1, 40]


def test_the_best_views_come_from_the_stage1_view_scores():
    view_score = np.array([[[0.1, 0.9, 0.5, 0.7], [0.4, 0.3, 0.2, 0.1]]])  # 1 photo, 2 candidates, 4 views
    assert best_views(view_score, top=2, views=2).tolist() == [[[1, 3], [0, 1]]]

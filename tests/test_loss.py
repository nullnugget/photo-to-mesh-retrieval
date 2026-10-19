"""Tests for src/train/losses.py on the worked example of the design notes.

Six meshes: chairs A, B, C (category 0), sofa D (1), table E (2), bed F (3). Three queries:
p1 (chair A), p2 (sofa D), p3 (chair A, from a difficult angle; chair B scores higher).
tau = 0.1, lambda = 0.3. Hand calculation, per query (L = L_inst + 0.3 × L_cat):

    p1: p(A) = 0.6345, L_inst 0.455, p(chair) 0.9832, L_cat 0.017, L 0.460
    p2: p(D) = 0.8934, L_inst 0.113, p(sofa)  0.8934, L_cat 0.113, L 0.147
    p3: p(A) = 0.3513, L_inst 1.046, p(chair) 0.9813, L_cat 0.019, L 1.052
    batch loss = (0.460 + 0.147 + 1.052) / 3 = 0.553
"""

import numpy as np
import pytest
import torch

from src.train.losses import gallery_info_nce, gallery_targets

SCORES = torch.tensor([
    [0.78, 0.70, 0.55, 0.40, 0.10, 0.20],  # p1, correct A
    [0.35, 0.30, 0.25, 0.72, 0.15, 0.45],  # p2, correct D
    [0.62, 0.66, 0.50, 0.30, 0.05, 0.15],  # p3, correct A
], dtype=torch.float64)
MESHES = ["A", "B", "C", "D", "E", "F"]
MESH_CATEGORIES = torch.tensor([0, 0, 0, 1, 2, 3])
QUERY_CATEGORIES = torch.tensor([0, 1, 0])
TARGETS = gallery_targets(["A", "D", "A"], MESHES)


def _loss(scores, category_weight=0.3):
    return gallery_info_nce(scores, TARGETS, QUERY_CATEGORIES, MESH_CATEGORIES, temperature=0.1, category_weight=category_weight)


def test_the_worked_example_gives_0_553():
    result = _loss(SCORES)
    assert TARGETS.tolist() == [0, 3, 0]
    assert round(result["loss"].item(), 3) == 0.553
    assert round(result["instance"].item(), 3) == round((0.455 + 0.113 + 1.046) / 3, 3)
    assert round(result["category"].item(), 3) == round((0.017 + 0.113 + 0.019) / 3, 3)


def test_each_query_matches_the_hand_calculation():
    for row, (inst, cat) in enumerate([(0.455, 0.017), (0.113, 0.113), (1.046, 0.019)]):
        one = gallery_info_nce(SCORES[row : row + 1], TARGETS[row : row + 1], QUERY_CATEGORIES[row : row + 1], MESH_CATEGORIES, 0.1, 0.3)
        assert round(one["instance"].item(), 3) == inst and round(one["category"].item(), 3) == cat


def test_the_pushes_on_the_scores_of_p3():
    # d L / d score = push / (tau × batch size); push = p(m) − [m is correct] for L_inst.
    scores = SCORES.clone().requires_grad_(True)
    _loss(scores, category_weight=0.0)["loss"].backward()
    push = (scores.grad[2] * 0.1 * 3).numpy()
    assert np.allclose(push, [-0.6487, 0.5241, 0.1058, 0.0143, 0.0012, 0.0032], atol=1e-3)
    # The category loss alone: the chairs together are pulled up, the other categories pushed down.
    scores = SCORES.clone().requires_grad_(True)
    gallery_info_nce(scores, TARGETS, QUERY_CATEGORIES, MESH_CATEGORIES, 0.1, 1.0)["category"].backward()
    assert np.allclose((scores.grad[2] * 0.1 * 3).numpy(), [-0.0067, -0.0100, -0.0020, 0.0143, 0.0012, 0.0032], atol=1e-3)


def test_category_weight_zero_is_the_instance_loss():
    result = _loss(SCORES, category_weight=0.0)
    assert torch.isclose(result["loss"], result["instance"])


def test_large_scores_do_not_overflow():
    result = gallery_info_nce(SCORES * 1000, TARGETS, QUERY_CATEGORIES, MESH_CATEGORIES, 0.01, 0.3)
    assert torch.isfinite(result["loss"])


def test_a_mesh_cannot_appear_twice_in_the_gallery():
    with pytest.raises(ValueError):
        gallery_targets(["A"], ["A", "B", "A"])
    with pytest.raises(KeyError):
        gallery_targets(["Z"], MESHES)

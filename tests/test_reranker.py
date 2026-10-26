"""Tests for re-ranking method 2: the pair scorer and its training lists (CPU, small sizes)."""

import numpy as np
import torch

from src.models.reranker import PairScorer, final_scores, select_tokens
from src.pipeline.train_reranker import training_list


def _pair(n=3, length=6, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = lambda: torch.randn(n, length, dim, generator=g)
    position = torch.randint(0, 28 * 28, (n, length), generator=g)
    mask = torch.ones(n, length, dtype=torch.bool)
    return x(), position, mask, x(), position.clone(), mask.clone()


def test_the_correction_is_zero_before_training():
    model = PairScorer(in_dim=8, width=16, heads=2, blocks=1, ffn=32)
    assert torch.equal(model(*_pair()), torch.zeros(3))


def test_padding_tokens_do_not_change_the_correction():
    torch.manual_seed(0)
    model = PairScorer(in_dim=8, width=16, heads=2, blocks=2, ffn=32)
    torch.nn.init.normal_(model.score[-1].weight)  # a non-zero output, so that a change would show
    p, pp, pm, v, vp, vm = _pair(n=1)
    base = model(p, pp, pm, v, vp, vm)
    pad = lambda x, value: torch.cat([x, value], 1)
    padded = model(pad(p, torch.randn(1, 3, 8)), pad(pp, torch.zeros(1, 3, dtype=torch.long)), pad(pm, torch.zeros(1, 3, dtype=torch.bool)),
                   pad(v, torch.randn(1, 2, 8)), pad(vp, torch.zeros(1, 2, dtype=torch.long)), pad(vm, torch.zeros(1, 2, dtype=torch.bool)))
    assert torch.allclose(base, padded, atol=1e-5)


def test_the_place_of_a_patch_changes_the_correction():
    torch.manual_seed(0)
    model = PairScorer(in_dim=8, width=16, heads=2, blocks=1, ffn=32)
    torch.nn.init.normal_(model.score[-1].weight)
    p, pp, pm, v, vp, vm = _pair(n=1)
    assert not torch.allclose(model(p, pp, pm, v, vp, vm), model(p, (pp + 1) % 784, pm, v, vp, vm))


def test_tokens_are_the_object_patches_first():
    vectors = torch.arange(10.0).view(1, 10, 1)
    on_object = torch.tensor([[False, True, False, True, True, False, False, False, False, False]])
    tokens, position, mask = select_tokens(vectors, on_object, 4, torch.Generator().manual_seed(0))
    assert set(position[0][mask[0]].tolist()) == {1, 3, 4} and mask.sum() == 3
    tokens, position, mask = select_tokens(vectors, torch.zeros(1, 10, dtype=torch.bool), 4)
    assert mask.all()  # no object patch: all patches count


def test_training_list_keeps_allowed_meshes_and_adds_the_correct_one():
    top = np.array([7, 3, 9, 1, 5, 2, 8, 0, 6, 4])  # stored list, best first
    allowed = np.ones(10, dtype=bool); allowed[[3, 1]] = False  # meshes 3 and 1 are held out (rule R2)
    chosen, target = training_list(top, allowed, truth=4, top=3)
    assert top[chosen].tolist() == [7, 9, 5, 4] and target == 3  # the first 3 allowed, then the correct mesh
    chosen, target = training_list(top, allowed, truth=9, top=3)
    assert top[chosen].tolist() == [7, 9, 5, 2] and target == 1  # the correct mesh is inside: the next allowed one is added
    assert training_list(top, allowed, truth=11, top=3) is None  # not in the stored list


def test_final_scores_of_the_two_variants():
    s1, m1, r = torch.tensor([[1.0, 2.0]]), torch.tensor([[0.5, 0.1]]), torch.tensor([[0.2, 0.0]])
    assert torch.allclose(final_scores("residual_method1", s1, m1, r, 1.0), torch.tensor([[1.7, 2.1]]))
    assert torch.allclose(final_scores("stage1", s1, m1, r, 1.0), torch.tensor([[1.2, 2.0]]))

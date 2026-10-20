"""Tests for src/train/trainer.py and the worker setup of src/data/datasets.py (CPU, synthetic features)."""

import os

import pytest
import torch

from src.config import load_config
from src.data.datasets import loader_worker_init
from src.models.embedding_heads import EmbeddingHeads
from src.retrieval import metrics
from src.train import trainer

TRAIN_CFG = load_config(["train.yaml"])["train"]


def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def _toy(meshes=6, views=5, dim=32, seed=0):
    """A gallery of random meshes, and queries that are noisy copies of one view of their mesh."""
    g = torch.Generator().manual_seed(seed)
    views_t = _unit(torch.randn(meshes, views, dim, generator=g))
    gallery = trainer.Gallery(views=views_t, mesh_ids=[f"m{i}" for i in range(meshes)], categories=torch.arange(meshes) % 2)
    targets = torch.arange(meshes).repeat(4)
    queries = _unit(views_t[targets, torch.randint(0, views, (len(targets),), generator=g)] + 0.3 * torch.randn(len(targets), dim, generator=g))
    return gallery, queries, targets


def test_learning_rate_rises_then_falls_along_a_cosine():
    factors = [trainer.learning_rate_factor(step, 10, 110) for step in range(110)]
    assert factors[0] == pytest.approx(0.1) and factors[9] == pytest.approx(1.0)
    assert factors[10] == pytest.approx(1.0) and factors[60] == pytest.approx(0.5)
    assert factors[-1] < 0.001 and all(a >= b for a, b in zip(factors[10:], factors[11:]))


def test_weight_decay_only_on_weight_matrices():
    heads = EmbeddingHeads(32, hidden_dim=16, embedding_dim=8, attention_dim=4)
    decay, no_decay = trainer.parameter_groups(heads, 1e-4)
    assert decay["weight_decay"] == 1e-4 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim == 2 for p in decay["params"]) and all(p.ndim == 1 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(heads.parameters()))


def test_the_training_gallery_keeps_only_the_meshes_of_the_training_photos():
    gallery, _, _ = _toy()
    kept = trainer.training_gallery({"toy": gallery}, {"m1", "m4"})
    assert kept.mesh_ids == ["m1", "m4"] and kept.views.shape[0] == 2 and kept.categories.tolist() == [1, 0]


def test_training_steps_lower_the_loss_and_change_only_the_heads():
    gallery, queries, targets = _toy()
    torch.manual_seed(0)
    heads = EmbeddingHeads(32, hidden_dim=64, embedding_dim=16, attention_dim=8)
    cfg = {**TRAIN_CFG, "heads": {**TRAIN_CFG["heads"], "view_dropout_min_views": 3}}
    optimizer = torch.optim.AdamW(trainer.parameter_groups(heads, 1e-4), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    views_before = gallery.views.clone()
    g = torch.Generator().manual_seed(0)
    losses = [trainer.train_step(heads, optimizer, scheduler, queries, targets, targets % 2, gallery, cfg, g)["loss"] for _ in range(60)]
    assert losses[-1] < 0.5 * losses[0]
    assert torch.equal(gallery.views, views_before)  # the stored features never change


def test_validation_matches_the_metric_functions():
    gallery, queries, targets = _toy()
    torch.manual_seed(1)
    heads = EmbeddingHeads(32, hidden_dim=64, embedding_dim=16, attention_dim=8)
    zero_shot = queries @ gallery.views.mean(dim=1).T  # any fixed scores
    val = trainer.ValSet("toy_queries_sam", "toy", queries, targets, targets % 2, zero_shot_scores=zero_shot, zero_shot_r1=0.0)
    result = trainer.validate(heads, [val], {"toy": gallery}, alpha=0.5)["toy_queries_sam"]
    with torch.no_grad():
        q = heads.embed_queries(queries)
        views, meshes = heads.embed_meshes(gallery.views)
        expected = metrics.correct_ranks((0.5 * torch.einsum("qd,mvd->qmv", q, views).amax(2) + 0.5 * q @ meshes.T).numpy(), targets.numpy())
    assert result["R@1"] == pytest.approx(metrics.recall_at_k(expected, 1)) and result["photos"] == len(targets)
    assert result["trained_R@1"] == result["R@1"]  # mix_weight 1.0: the trained score alone
    only_zero_shot = trainer.validate(heads, [val], {"toy": gallery}, alpha=0.5, mix_weight=0.0)["toy_queries_sam"]
    assert only_zero_shot["R@1"] == pytest.approx(metrics.recall_at_k(metrics.correct_ranks(zero_shot.numpy(), targets.numpy()), 1))
    half = trainer.validate(heads, [val], {"toy": gallery}, alpha=0.5, mix_weight=0.5)["toy_queries_sam"]
    mixed = 0.5 * (0.5 * torch.einsum("qd,mvd->qmv", q, views).amax(2) + 0.5 * q @ meshes.T) + 0.5 * zero_shot
    assert half["R@1"] == pytest.approx(metrics.recall_at_k(metrics.correct_ranks(mixed.numpy(), targets.numpy()), 1))
    assert trainer.selection_value({"a": {"R@1": 0.6}, "b": {"R@1": 0.4}}) == pytest.approx(0.5)
    assert heads.training  # validation puts the heads back into training mode


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="CPU binding is Linux only")
def test_a_worker_gets_all_cores_and_one_thread():
    threads, cores = torch.get_num_threads(), os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, {min(cores)})  # imitate a worker that inherited a binding to one CPU
        loader_worker_init(0)
        assert len(os.sched_getaffinity(0)) == os.cpu_count() and torch.get_num_threads() == 1
    finally:
        os.sched_setaffinity(0, cores)
        torch.set_num_threads(threads)

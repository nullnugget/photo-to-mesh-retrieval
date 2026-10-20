"""Tests for src/models/embedding_heads.py: shapes, unit length, view masks, parameters, and scores."""

import numpy as np
import pytest
import torch

from src.config import load_config
from src.models.embedding_heads import EmbeddingHeads, ViewAggregator, parameter_counts, random_view_mask, scores
from src.retrieval import search

torch.manual_seed(0)
HEADS = EmbeddingHeads(feature_dim=768)


def test_shapes_and_unit_length():
    q = HEADS.embed_queries(torch.randn(5, 768))
    views, meshes = HEADS.embed_meshes(torch.randn(3, 25, 768))
    assert q.shape == (5, 256) and views.shape == (3, 25, 256) and meshes.shape == (3, 256)
    for embedding in (q, views, meshes):
        assert torch.allclose(embedding.norm(dim=-1), torch.ones(embedding.shape[:-1]), atol=1e-4)


def test_parameter_counts():
    counts = parameter_counts(HEADS)
    projector = 768 * 512 + 512 + 512 * 256 + 256  # two linear layers with biases
    assert counts["query_projector"] == counts["view_projector"] == projector == 525_056
    assert counts["view_aggregator"] == 256 * 128 + 128 + 128 == 33_024  # rating layer (weights, biases) and rating vector
    assert counts["total"] == 2 * projector + 33_024 == 1_083_136
    assert parameter_counts(EmbeddingHeads(768, aggregator="mean"))["view_aggregator"] == 0


def test_from_config_uses_the_training_settings():
    heads = EmbeddingHeads.from_config(load_config(["train.yaml"])["train"]["heads"], feature_dim=768)
    assert heads.embed_queries(torch.randn(2, 768)).shape == (2, 256)


def test_the_rating_is_tanh_attention():
    aggregator = ViewAggregator(256, "attention")
    torch.nn.init.normal_(aggregator.rating)
    views = torch.randn(3, 25, 256)
    ratings = torch.tanh(aggregator.key(views)) @ aggregator.rating  # (3, 25, 128) @ (128,) -> (3, 25)
    assert torch.allclose(aggregator.weights(views), torch.softmax(ratings, dim=1), atol=1e-6)


def test_the_attention_starts_as_the_plain_mean():
    views = torch.nn.functional.normalize(torch.randn(4, 25, 256), dim=-1)
    attention, mean = ViewAggregator(256, "attention"), ViewAggregator(256, "mean")
    assert torch.allclose(attention(views), mean(views), atol=1e-6)
    expected = torch.nn.functional.normalize(views.mean(dim=1), dim=-1)
    assert torch.allclose(mean(views), expected, atol=1e-6)


def test_masked_views_have_no_effect():
    aggregator = ViewAggregator(256, "attention")
    torch.nn.init.normal_(aggregator.rating)  # trained-like weights, not the zero start
    views = torch.randn(2, 25, 256)
    mask = torch.zeros(2, 25, dtype=torch.bool)
    mask[:, :12] = True
    changed = views.clone()
    changed[:, 12:] = torch.randn(2, 13, 256)  # change only the dropped views
    assert torch.allclose(aggregator(views, mask), aggregator(changed, mask), atol=1e-6)
    assert torch.allclose(aggregator.weights(views, mask)[:, 12:], torch.zeros(2, 13))


def test_random_view_mask_keeps_12_to_25_views():
    mask = random_view_mask(1000, 25, 12, generator=torch.Generator().manual_seed(0))
    kept = mask.sum(dim=1)
    assert mask.shape == (1000, 25) and kept.min() == 12 and kept.max() == 25
    assert mask.float().mean(dim=0).std() < 0.05  # every view position is dropped about equally often
    with pytest.raises(ValueError):
        random_view_mask(1, 25, 0)


def test_scores_equal_the_zero_shot_search():
    q = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    views = torch.nn.functional.normalize(torch.randn(4, 5, 16), dim=-1)
    meshes = torch.nn.functional.normalize(views.mean(dim=1), dim=-1)
    expected = search.combined_scores(q.numpy(), views.numpy(), 0.75)
    assert torch.allclose(scores(q, views, meshes, 0.75), torch.from_numpy(expected), atol=1e-5)


def test_the_gradient_reaches_all_three_heads():
    heads = EmbeddingHeads(768)
    torch.nn.init.normal_(heads.view_aggregator.rating)  # at zero, the key layer gets no gradient yet
    q = heads.embed_queries(torch.randn(4, 768))
    views, meshes = heads.embed_meshes(torch.randn(3, 25, 768), random_view_mask(3, 25, 12))
    scores(q, views, meshes, 0.5).sum().backward()
    for name in ("query_projector", "view_projector", "view_aggregator"):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(heads, name).parameters()), name


@pytest.mark.parametrize("design", ["residual", "query_residual"])
def test_the_residual_designs_start_as_the_zero_shot_scores(design):
    torch.manual_seed(0)
    heads = EmbeddingHeads(32, hidden_dim=16, attention_dim=8, design=design)
    raw_views = torch.nn.functional.normalize(torch.randn(5, 4, 32), dim=-1)
    raw_queries = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
    q = heads.embed_queries(raw_queries)
    views, meshes = heads.embed_meshes(raw_views)
    assert q.shape == (3, 32) and views.shape == (5, 4, 32) and heads.embedding_dim == 32  # the feature length stays
    assert torch.allclose(q, raw_queries, atol=1e-6) and torch.allclose(views, raw_views, atol=1e-6)
    zero_shot = search.combined_scores(raw_queries.numpy(), raw_views.numpy(), 0.5)
    assert np.allclose(scores(q, views, meshes, 0.5).detach().numpy(), zero_shot, atol=1e-5)


def test_the_designs_train_different_parts():
    mlp = 768 * 512 + 512 + 512 * 768 + 768  # 787,712
    residual = parameter_counts(EmbeddingHeads(768, design="residual"))
    assert residual["query_projector"] == residual["view_projector"] == mlp and residual["view_aggregator"] == 768 * 128 + 128 + 128
    query_only = parameter_counts(EmbeddingHeads(768, design="query_residual"))
    assert query_only == {"query_projector": mlp, "view_projector": 0, "view_aggregator": 0, "total": mlp}
    assert EmbeddingHeads.from_config({"hidden_dim": 512, "embedding_dim": 256, "aggregator": "attention", "attention_dim": 128}, 768).design == "projection"
    with pytest.raises(ValueError):
        EmbeddingHeads(768, design="other")

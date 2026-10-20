"""Tests for src/retrieval/index.py, the score-matrix evaluation, and the locks of the trained evaluation (CPU)."""

import numpy as np
import pytest
import torch

from src import runs
from src.config import load_config
from src.models.embedding_heads import EmbeddingHeads, scores
from src.models.feature_store import QUERIES, RENDERS, FeatureTable, write_features
from src.pipeline import run_evaluation
from src.retrieval import index as gallery_index
from src.retrieval.evaluate import Gallery, Queries, evaluate, evaluate_scores, score_meshes

ZERO_SHOT = {"scoring": "combined", "alpha": 0.75}


def _unit_rows(rng, *shape):
    vectors = rng.normal(size=shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _renders(meshes=5, views=4, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    parts = {name: _unit_rows(rng, meshes, views, dim).astype(np.float16) for name in ("cls", "masked_mean")}
    labels = {"mesh_id": np.array([f"toy/chair/m{i}" for i in range(meshes)]), "category": np.array(["chair", "table"] * meshes)[:meshes]}
    return FeatureTable(RENDERS, parts, labels, {"views": views})


def _heads(dim=16, seed=0):
    torch.manual_seed(seed)
    heads = EmbeddingHeads(dim, hidden_dim=32, embedding_dim=8, attention_dim=4).eval()
    torch.nn.init.normal_(heads.view_aggregator.rating, std=1.0)  # not the plain mean, so the test sees the aggregator
    return heads


def _checkpoint(heads, heads_cfg):
    meta = {"config": {"train": {"heads": heads_cfg}}}
    return {**{key: getattr(heads, key).state_dict() for key in runs.HEAD_KEYS}, "meta": meta}


def _built(tmp_path):
    renders = _renders()
    renders_file = write_features(renders, tmp_path / "renders.npz")
    heads = _heads()
    info = {"renders_file": str(renders_file), "renders_sha256": runs.file_sha256(renders_file), "recipe": "cls"}
    return heads, gallery_index.build_index(heads, renders, "cls", info), renders_file


def test_the_index_holds_the_embeddings_of_the_heads_and_reads_back(tmp_path):
    heads, index, _ = _built(tmp_path)
    with torch.no_grad():
        views, meshes = heads.embed_meshes(torch.from_numpy(_renders().vectors("cls")).half())
    assert np.allclose(index.views, views.numpy()) and np.allclose(index.meshes, meshes.numpy())
    path = gallery_index.save_index(index, tmp_path / "embedding_heads_best_gallery_toy.npz")
    loaded = gallery_index.load_index(path)
    assert np.array_equal(loaded.views, index.views) and np.array_equal(loaded.raw_views, index.raw_views)
    assert loaded.mesh_ids.tolist() == index.mesh_ids.tolist() and loaded.info == index.info


def test_an_index_stops_when_its_renders_file_changes(tmp_path):
    _, index, renders_file = _built(tmp_path)
    path = gallery_index.save_index(index, tmp_path / "index.npz")
    write_features(_renders(seed=1), renders_file)  # new features under the same name
    with pytest.raises(gallery_index.IndexMismatch):
        gallery_index.load_index(path)


def test_the_mix_weight_moves_between_the_trained_and_the_zero_shot_score(tmp_path):
    heads, index, _ = _built(tmp_path)
    raw = _unit_rows(np.random.default_rng(5), 6, 16)
    search = gallery_index.search_scores(index, heads, raw, alpha=0.5, zero_shot=ZERO_SHOT)
    with torch.no_grad():
        expected = scores(heads.embed_queries(torch.from_numpy(raw).half()), torch.from_numpy(index.views),
                          torch.from_numpy(index.meshes), 0.5).numpy()
    assert np.allclose(gallery_index.mix_scores(search, 1.0), expected)
    assert np.allclose(gallery_index.mix_scores(search, 0.0), score_meshes(raw, index.raw_views, "combined", 0.75))
    assert np.allclose(gallery_index.mix_scores(search, 0.5), 0.5 * expected + 0.5 * search["zero_shot"])


def test_heads_load_from_a_checkpoint_with_the_feature_length_of_its_weights():
    heads = _heads()
    loaded = gallery_index.load_heads(_checkpoint(heads, {"hidden_dim": 32, "embedding_dim": 8, "aggregator": "attention", "attention_dim": 4}))
    x = torch.randn(3, 16)
    assert torch.equal(loaded.embed_queries(x), heads.embed_queries(x)) and not loaded.training


def test_metrics_from_a_score_matrix_equal_the_zero_shot_evaluation():
    renders = _renders()
    gallery = Gallery(renders.vectors("cls"), renders.ids, renders.labels["category"])
    rng = np.random.default_rng(2)
    targets = rng.integers(0, 5, 12)
    queries = Queries(_unit_rows(rng, 12, 16), np.array([f"q{i}" for i in range(12)]), renders.ids[targets], renders.labels["category"][targets])
    direct = evaluate(queries, gallery, "combined", 0.75)
    from_scores = evaluate_scores(score_meshes(queries.vectors, gallery.views, "combined", 0.75), queries, gallery.mesh_ids, gallery.categories)
    assert direct == from_scores


def test_the_trained_evaluation_keeps_the_locks(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(run_evaluation, "training_record", lambda run, mini=False: {"split": "seen_meshes", "datasets": ["pix3d"]})
    with pytest.raises(run_evaluation.TestPhotosLocked):
        run_evaluation.run_trained("pix3d", cfg, "toy_run", photos="test")
    with pytest.raises(ValueError, match="test-free"):
        run_evaluation.run_trained("pix3d", cfg, "toy_run", photos="test_free")
    with pytest.raises(ValueError, match="training photos"):  # photos of another split can be training photos
        run_evaluation.run_trained("pix3d", cfg, "toy_run", photos="val", split="unseen_meshes")

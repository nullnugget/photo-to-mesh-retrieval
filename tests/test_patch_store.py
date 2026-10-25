"""Tests for src/models/patch_store.py: the PCA of the patch vectors and the store files."""

import numpy as np
import pytest
import torch

from src.models.feature_store import QUERIES, RENDERS
from src.models.patch_store import CovarianceSum, PatchProjection, PatchWriter, patch_files, read_patches


def _correlated(rng, n=4000, dim=12):
    """Vectors with a clear order of variance along random directions, and an offset mean."""
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    scales = np.linspace(5.0, 0.1, dim)
    return (rng.normal(size=(n, dim)) * scales) @ basis.T + 3.0


def test_the_pca_matches_numpy():
    x = _correlated(np.random.default_rng(0))
    sums = CovarianceSum(x.shape[1])
    sums.add(torch.from_numpy(x[:1500])); sums.add(torch.from_numpy(x[1500:]))  # in two batches
    fit = sums.fit()
    variance, components = np.linalg.eigh(np.cov(x.T, bias=True))
    variance, components = variance[::-1], components[:, ::-1]
    assert fit.count == len(x)
    np.testing.assert_allclose(fit.mean, x.mean(0), rtol=1e-5)
    np.testing.assert_allclose(fit.variance, variance, rtol=1e-4)
    # each component equals numpy's up to its sign
    np.testing.assert_allclose(np.abs((fit.components * components).sum(0)), 1.0, atol=1e-4)


def test_the_projection_keeps_the_first_components_and_reconstructs_at_full_size():
    x = _correlated(np.random.default_rng(1))
    sums = CovarianceSum(x.shape[1]); sums.add(torch.from_numpy(x))
    fit = sums.fit()
    t = torch.from_numpy(x[:50]).float()
    full = fit.project(t, x.shape[1])
    np.testing.assert_allclose((full @ torch.from_numpy(fit.components).T + torch.from_numpy(fit.mean)).numpy(), x[:50], atol=1e-3)
    np.testing.assert_allclose(fit.project(t, 4).numpy(), full[:, :4].numpy(), atol=1e-5)  # 4 = the first 4 of 12
    assert 0.6 < fit.explained(4) < fit.explained(8) < fit.explained(12) == pytest.approx(1.0)


def test_a_projection_saves_and_loads(tmp_path):
    fit = PatchProjection(np.arange(3, dtype=np.float32), np.eye(3, dtype=np.float32), np.array([3, 2, 1], np.float32), 7)
    back = PatchProjection.load(fit.save(tmp_path / "p.npz"))
    assert back.count == 7 and np.array_equal(back.components, fit.components) and np.array_equal(back.mean, fit.mean)


@pytest.mark.parametrize("kind,shape", [(RENDERS, (3, 4)), (QUERIES, (5,))])
def test_a_written_set_reads_back(tmp_path, kind, shape):
    rng = np.random.default_rng(2)
    files = patch_files("bb", "pix3d", "renders" if kind == RENDERS else "queries_sam", tmp_path)
    images, patches, dim = int(np.prod(shape)), 6, 4
    vectors = rng.normal(size=(images, patches, dim)).astype(np.float32)
    shares = rng.uniform(size=(images, patches))
    writer = PatchWriter(files, shape, dim, patches)
    writer.add(vectors[:2], shares[:2]); writer.add(vectors[2:], shares[2:])
    assert not files["vectors"].exists()  # only the temporary name until close
    labels = ({"mesh_id": np.array(["a", "b", "c"]), "category": np.array(["x"] * 3)} if kind == RENDERS else
              {"query_id": np.array(list("abcde")), "mesh_id": np.array(["m"] * 5), "category": np.array(["x"] * 5)})
    writer.close(kind, labels, {"dim": dim})
    table = read_patches(files)
    assert table.kind == kind and table.info == {"dim": dim} and list(table.ids) == list(labels[next(iter(labels))])
    assert table.vectors.shape == (*shape, patches, dim) and table.object.shape == (*shape, patches)
    np.testing.assert_allclose(np.asarray(table.vectors, np.float32).reshape(images, patches, dim), vectors, atol=2e-3)
    np.testing.assert_allclose(np.asarray(table.object).reshape(images, patches) / 255, shares, atol=1 / 255)


def test_an_incomplete_set_is_not_closed(tmp_path):
    files = patch_files("bb", "abo", "queries_sam", tmp_path)
    writer = PatchWriter(files, (3,), 4, 6)
    writer.add(np.zeros((2, 6, 4)), np.zeros((2, 6)))
    with pytest.raises(ValueError, match="2 of 3"):
        writer.close(QUERIES, {"query_id": np.array(list("abc")), "mesh_id": np.array(list("mmm")), "category": np.array(list("xxx"))}, {})
    assert not files["vectors"].exists()

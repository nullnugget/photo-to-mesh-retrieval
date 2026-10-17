"""Tests for src/render/simplify.py."""

import torch

from src.render.simplify import merge_vertices_in_grid


def _fine_grid(n):
    """A flat square from -0.5 to 0.5 in x and y, split into n x n squares of 2 triangles each."""
    coords = torch.linspace(-0.5, 0.5, n + 1)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    vertices = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros((n + 1) ** 2)], dim=1)
    index = torch.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, :-1], index[1:, 1:]
    faces = torch.cat([torch.stack([a, b, d], -1).reshape(-1, 3), torch.stack([a, d, c], -1).reshape(-1, 3)])
    return vertices, faces


def test_fewer_faces_same_extent_no_collapsed_triangles():
    vertices, faces = _fine_grid(200)  # 80,000 triangles, 0.005 apart
    merged, new_faces = merge_vertices_in_grid(vertices, faces, cell_size=0.05)
    assert len(new_faces) < len(faces) / 50
    assert torch.allclose(merged.min(dim=0).values, vertices.min(dim=0).values, atol=0.05)  # shape kept within one cell
    assert torch.allclose(merged.max(dim=0).values, vertices.max(dim=0).values, atol=0.05)
    a, b, c = new_faces.unbind(dim=1)
    assert ((a != b) & (b != c) & (a != c)).all()


def test_tiny_cell_changes_nothing():
    vertices, faces = _fine_grid(10)
    merged, new_faces = merge_vertices_in_grid(vertices, faces, cell_size=1e-4)
    assert len(merged) == len(vertices) and len(new_faces) == len(faces)

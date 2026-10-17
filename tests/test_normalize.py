"""Tests for src/render/normalize.py and src/render/mesh_io.py."""

import math

import pytest
import torch

from src.render.mesh_io import EmptyMeshError, load_mesh
from src.render.normalize import DegenerateMeshError, normalize_to_unit_sphere, turn_to_front


def test_unit_sphere_and_centred_box():
    vertices = torch.tensor([[10.0, 0.0, 0.0], [14.0, 2.0, 1.0], [12.0, 6.0, -3.0]])
    normalized, centre, radius = normalize_to_unit_sphere(vertices)
    assert math.isclose(float(normalized.norm(dim=1).max()), 1.0, abs_tol=1e-6)
    box_centre = (normalized.min(dim=0).values + normalized.max(dim=0).values) / 2
    assert torch.allclose(box_centre, torch.zeros(3), atol=1e-6)
    assert torch.allclose(normalized * radius + centre, vertices, atol=1e-5)  # the change can be undone


def test_degenerate_mesh_raises():
    with pytest.raises(DegenerateMeshError):
        normalize_to_unit_sphere(torch.ones(4, 3))


@pytest.mark.parametrize("front", ["+z", "-z", "+x", "-x"])
def test_turn_brings_the_front_to_plus_z(front):
    sign = 1.0 if front[0] == "+" else -1.0
    axis = {"x": 0, "z": 2}[front[1]]
    point = torch.zeros(1, 3)
    point[0, axis] = sign  # a point on the front side of the mesh
    point[0, 1] = 0.5  # and above the centre
    turned = turn_to_front(point, front)
    assert torch.allclose(turned, torch.tensor([[0.0, 0.5, 1.0]]), atol=1e-6)


def test_turn_is_not_a_mirror():
    basis = torch.eye(3)
    for front in ["-z", "+x", "-x"]:
        assert math.isclose(float(torch.linalg.det(turn_to_front(basis, front))), 1.0, abs_tol=1e-6)


def test_load_obj_and_ply(tmp_path):
    from src.data.glb import write_ply

    (tmp_path / "quad.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")  # one 4-corner face
    vertices, faces = load_mesh(tmp_path / "quad.obj")
    assert vertices.shape == (4, 3) and faces.shape == (2, 3)  # split into 2 triangles
    write_ply(tmp_path / "quad.ply", vertices.numpy(), faces.numpy())
    vertices_ply, faces_ply = load_mesh(tmp_path / "quad.ply")
    assert torch.equal(vertices_ply, vertices) and torch.equal(faces_ply, faces)
    (tmp_path / "empty.obj").write_text("v 0 0 0\n")
    with pytest.raises(EmptyMeshError):
        load_mesh(tmp_path / "empty.obj")

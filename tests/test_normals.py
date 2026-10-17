"""Tests for src/render/normals.py, on a small box and a flipped triangle."""

import torch

from src.render.normals import corner_normals, face_normals

# A cube from 0 to 1, 12 triangles, all wound so that the normals point outwards.
CUBE_VERTICES = torch.tensor([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
CUBE_FACES = torch.tensor([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],  # x = 0 and x = 1
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],  # y = 0 and y = 1
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],  # z = 0 and z = 1
])


def test_face_normals_point_outwards():
    normals, areas = face_normals(CUBE_VERTICES, CUBE_FACES)
    centres = CUBE_VERTICES[CUBE_FACES].mean(dim=1) - 0.5
    assert ((normals * centres).sum(dim=1) > 0).all()
    assert torch.allclose(areas, torch.full((12,), 0.5))


def test_crease_keeps_box_edges_sharp_and_smooth_does_not():
    flat_normals, _ = face_normals(CUBE_VERTICES, CUBE_FACES)
    crease = corner_normals(CUBE_VERTICES, CUBE_FACES, "crease", 30.0)
    assert torch.allclose(crease, flat_normals[:, None, :].expand(-1, 3, -1), atol=1e-6)  # 90-degree edges stay sharp
    smooth = corner_normals(CUBE_VERTICES, CUBE_FACES, "smooth")
    assert not torch.allclose(smooth, crease, atol=1e-3)  # smooth blends across the edges


def test_crease_blends_a_shallow_fold_and_a_flipped_neighbour():
    # Two triangles that share the edge (0, 1) and meet at about 11 degrees; the second is wound backwards.
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.2]])
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]])  # the second triangle's normal points to -z
    normals, _ = face_normals(vertices, faces)
    assert (normals[0] * normals[1]).sum() < 0  # opposite sides, because of the winding
    corners = corner_normals(vertices, faces, "crease", 30.0)
    shared = corners[0, 0]  # the corner at vertex 0 of the first triangle
    assert torch.allclose(shared.norm(), torch.tensor(1.0))
    assert not torch.allclose(shared, normals[0], atol=1e-3)  # blended with the (turned) neighbour
    assert (shared * normals[0]).sum() > 0.95  # and still on the side of its own triangle

"""Tests for src/data/mesh_twins.py (synthetic meshes; no dataset)."""

import numpy as np
import torch
from pytorch3d.structures import Meshes

from src.data import mesh_twins

# A unit cube centred at the origin: 8 corners, 12 triangles.
CUBE_VERTICES = torch.tensor([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=torch.float32)
CUBE_FACES = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                           [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def _cube(scale=(1.0, 1.0, 1.0)):
    return Meshes([CUBE_VERTICES * torch.tensor(scale)], [CUBE_FACES])


def test_the_grid_marks_the_cells_of_the_points_and_grows_by_one():
    points = torch.tensor([[-1.0, -1.0, -1.0], [0.99, 0.99, 0.99], [0.0, 0.0, 0.0]])
    grid = mesh_twins.occupancy_grid(points, cells=4, dilate=False)
    assert grid.sum() == 3 and grid[0, 0, 0] and grid[3, 3, 3] and grid[2, 2, 2]
    grown = mesh_twins.occupancy_grid(points[:1], cells=4, dilate=True)
    assert grown.sum() == 8 and grown[:2, :2, :2].all()  # a corner cell grows into its 2 × 2 × 2 neighbourhood


def test_grid_iou_counts_shared_cells():
    a = np.zeros((2, 2, 2, 2), dtype=bool)
    a[0].flat[[0, 1, 2]] = True  # cells {0, 1, 2}
    a[1].flat[[1, 2, 3]] = True  # cells {1, 2, 3}
    iou = mesh_twins.grid_iou(a)
    assert np.allclose(np.diag(iou), 1.0) and np.isclose(iou[0, 1], 2 / 4) and np.isclose(iou[1, 0], iou[0, 1])
    assert np.allclose(mesh_twins.grid_iou(a, chunk=1), iou)


def test_the_surface_distance_is_zero_for_the_same_form_and_grows_with_the_difference():
    same = [mesh_twins.surface_distance(_cube(), _cube(), points=n) for n in (100, 4000)]
    other = mesh_twins.surface_distance(_cube(), _cube((1.0, 0.6, 1.0)), points=4000)
    assert max(same) < 1e-4  # points on a surface lie on it: no sampling floor, also with 100 points
    assert other > 0.05


def test_grids_survive_saving(tmp_path):
    grids = np.random.default_rng(0).random((3, 8, 8, 8)) > 0.7
    path = mesh_twins.save_grids(tmp_path / "grids.npz", ["a", "b", "c"], grids)
    ids, loaded = mesh_twins.load_grids(path)
    assert ids == ["a", "b", "c"] and np.array_equal(loaded, grids)


def test_point_samples_repeat_with_the_seed_and_leave_the_global_generator_alone():
    torch.manual_seed(5)
    expected = torch.rand(1)
    torch.manual_seed(5)
    first = mesh_twins.surface_points(_cube(), 100, seed=3)
    assert torch.equal(torch.rand(1), expected)
    assert torch.equal(first, mesh_twins.surface_points(_cube(), 100, seed=3))

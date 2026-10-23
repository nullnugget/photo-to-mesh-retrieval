"""Near-identical meshes ("twins"): find them by geometry alone.

Many ABO products are colour variants of one form, sold under their own product ids. Their mesh
files differ a little (another triangle list, small rounding), so the merge of identical geometry
(`src/pipeline/catalog_abo_meshes.py`) does not join them. The renders are grey, so two twins give
the same renders, and a search cannot tell them apart: the "wrong" twin counts as an error by
chance, and the training pushes apart two meshes that look the same.

The decision "the same form" uses geometry only, never the backbone features: the backbone is the
part that is judged, so it must not also be the judge. Two stages, because the exact test is slow:

1. Occupancy grid (fast, all pairs). Each mesh is scaled to the unit sphere (as for the renders;
   all meshes share the axes +Y up, +Z front). Points are sampled on its surface (more on larger
   triangles), and each point marks its cell of a cells³ grid: a coarse shell of the surface. The
   grid grows by one cell in every direction (dilation), so that a copy shifted by half a cell
   still overlaps. The overlap of two grids is IoU = cells of both / cells of either. With all
   grids as rows of a matrix, one matrix product gives the intersection of every pair.

       2-D example, 4 × 4 cells:  A = {1, 2, 3, 4, 5}, B = {2, 3, 4, 5, 6}
       intersection 4, union 6, IoU 0.67

2. Surface distance (exact, candidate pairs only). Points sampled on mesh A, their distance to the
   nearest triangle of mesh B, and the other way; the root of the mean squared distance, in units of
   the sphere radius. Only the point-to-triangle direction is used: a point that lies on a surface
   is at distance 0 from it, whatever the number of points, so the measure has no sampling floor.
   (Point to point has one: two random samples of one surface are about 0.025 apart at 5,000
   points. So has triangle to point, the second half of `pytorch3d.loss.point_mesh_face_distance`:
   it depends on how densely the points cover each triangle.)

The size is not in the geometry test (the unit sphere hides it): a 30-inch and a 40-inch version of
one design pass both stages. The listing dimensions separate them (a later step).
"""

from pathlib import Path

import numpy as np
import torch
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from src.render.mesh_io import load_mesh
from src.render.normalize import normalize_to_unit_sphere

GRID_CELLS = 32  # cells per side of the occupancy grid
GRID_POINTS = 50_000  # surface points per grid: enough to mark every surface cell
DISTANCE_POINTS = 5_000  # surface points per side of the exact test (cost: points × triangles; 2,000 to 20,000 give the same values ± 0.0001)


def unit_mesh(path: str | Path, device: str | torch.device = "cpu") -> Meshes:
    """Read a mesh and scale it to the unit sphere (bounding-box centre at the origin, farthest vertex at 1)."""
    vertices, faces = load_mesh(path)
    vertices = normalize_to_unit_sphere(vertices)[0].float()
    return Meshes([vertices.to(device)], [faces.to(device)])


def surface_points(mesh: Meshes, points: int, seed: int = 0) -> torch.Tensor:
    """Sample points on the surface (area-weighted); the seed makes the sample repeatable. Returns (points, 3)."""
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        return sample_points_from_meshes(mesh, points)[0]
    finally:
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def occupancy_grid(points: torch.Tensor, cells: int = GRID_CELLS, dilate: bool = True) -> np.ndarray:
    """Return the cells³ grid of surface points in the unit sphere: bool (cells, cells, cells), axes x, y, z.

    Args:
        points: (N, 3) points with coordinates in −1 to 1.
        cells: Cells per side.
        dilate: Grow the filled cells by one cell in every direction (also the diagonals).
    """
    index = ((points.detach().cpu().numpy() + 1.0) / 2.0 * cells).astype(np.int64).clip(0, cells - 1)
    grid = np.zeros((cells, cells, cells), dtype=bool)
    grid[index[:, 0], index[:, 1], index[:, 2]] = True
    if dilate:
        grown = grid.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    grown |= _shift(grid, dx, dy, dz)
        grid = grown
    return grid


def _shift(grid: np.ndarray, dx: int, dy: int, dz: int) -> np.ndarray:
    """The grid moved by (dx, dy, dz) cells; cells that come in from outside are empty."""
    out = np.zeros_like(grid)
    n = grid.shape[0]
    src = [slice(max(0, -d), n - max(0, d)) for d in (dx, dy, dz)]
    dst = [slice(max(0, d), n - max(0, -d)) for d in (dx, dy, dz)]
    out[dst[0], dst[1], dst[2]] = grid[src[0], src[1], src[2]]
    return out


def grid_iou(grids_a: np.ndarray, grids_b: np.ndarray | None = None, device: str | torch.device = "cpu",
             chunk: int = 512) -> np.ndarray:
    """IoU of every pair of grids: float32 (A, B).

    Args:
        grids_a, grids_b: bool (count, cells, cells, cells); None for B = A.
        device: Where to multiply (the intersections of all pairs are one matrix product).
        chunk: Rows of A per product (bounded memory).

    Intersections are counted in 32-bit floats: half precision cannot hold every count above 2,048
    exactly.
    """
    grids_b = grids_a if grids_b is None else grids_b
    b = torch.from_numpy(grids_b.reshape(len(grids_b), -1)).to(device, torch.float32)
    a_all = grids_a.reshape(len(grids_a), -1)
    size_b = b.sum(dim=1)
    out = np.empty((len(a_all), len(b)), dtype=np.float32)
    for i in range(0, len(a_all), chunk):
        a = torch.from_numpy(a_all[i:i + chunk]).to(device, torch.float32)
        inter = a @ b.T
        union = a.sum(dim=1, keepdim=True) + size_b[None, :] - inter
        out[i:i + chunk] = (inter / union.clamp(min=1)).cpu().numpy()
    return out


MIN_TRIANGLE_AREA = 1e-10  # pytorch3d measures to the edges of smaller triangles (its default 0.005 is coarse for unit-sphere meshes)


def point_to_surface(points: torch.Tensor, mesh: Meshes) -> torch.Tensor:
    """Squared distance of each point (P, 3) to the nearest triangle of a single mesh: (P,)."""
    tris = mesh.verts_packed()[mesh.faces_packed()]  # (T, 3, 3)
    zero = torch.zeros(1, dtype=torch.int64, device=points.device)
    return point_face_distance(points, zero, tris, zero, len(points), MIN_TRIANGLE_AREA)


def surface_distance(mesh_a: Meshes, mesh_b: Meshes, points: int = DISTANCE_POINTS, seed: int = 0) -> float:
    """Root of the mean squared point-to-surface distance, both ways, in unit-sphere units (0 for the same surface)."""
    pa, pb = surface_points(mesh_a, points, seed), surface_points(mesh_b, points, seed + 1)
    squared = 0.5 * (float(point_to_surface(pa, mesh_b).mean()) + float(point_to_surface(pb, mesh_a).mean()))
    return squared ** 0.5


def save_grids(path: Path, mesh_ids: list[str], grids: np.ndarray) -> Path:
    """Write the grids as packed bits (cells³ / 8 bytes per mesh) with their mesh ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mesh_ids=np.array(mesh_ids), cells=grids.shape[1],
                        bits=np.packbits(grids.reshape(len(grids), -1), axis=1))
    return path


def load_grids(path: Path) -> tuple[list[str], np.ndarray]:
    """Read the grids of `save_grids`: (mesh ids, bool (count, cells, cells, cells))."""
    with np.load(path) as data:
        cells = int(data["cells"])
        grids = np.unpackbits(data["bits"], axis=1, count=cells ** 3).astype(bool).reshape(-1, cells, cells, cells)
        return data["mesh_ids"].tolist(), grids

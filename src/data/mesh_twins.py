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

   The distance to a triangle is calculated here with the closest-point method of Ericson
   ("Real-Time Collision Detection", 5.1.5): the point's closest place on the triangle is a corner,
   a point on an edge, or a point inside, decided by dot products. Why not
   `pytorch3d.loss.point_mesh_distance.point_face_distance`: with a small `min_triangle_area` it
   returned values about 100 times too small for a closed cabinet against an open shelf (points on a
   side panel got distance about 0 from the coplanar faces of thin corner posts), and with its
   default (0.005) it measures to the edges of every smaller triangle, which overstates the
   distance. A test compares this method with a dense brute-force sample.

The size is not in the geometry test (the unit sphere hides it): a 30-inch and a 40-inch version of
one design pass both stages. The listing dimensions separate them (a later step).
"""

from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from src.render.mesh_io import load_mesh
from src.render.normalize import normalize_to_unit_sphere

GRID_CELLS = 32  # cells per side of the occupancy grid
GRID_POINTS = 50_000  # surface points per grid: enough to mark every surface cell
DISTANCE_POINTS = 2_000  # surface points per side of the exact test (cost: points × triangles; no sampling floor, so few points suffice)
BOUND_POINTS = 50_000  # dense surface sample of the fast upper bound (`distance_bound`)


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


POINT_TRIANGLE_BLOCK = 2**23  # point × triangle pairs per block in `point_to_surface`: about 1.2 GiB of GPU memory; 2**25 needs 4.3 GiB at the same speed


def point_triangle_sq_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Squared distance of each point to each triangle: p (P, 3); a, b, c (T, 3) corners → (P, T).

    The closest place on a triangle is found by the regions of Ericson's method (corner a, b, c;
    edge ab, ac, bc; or inside). A zero-area triangle falls into a corner or edge region.
    """
    p = p[:, None, :]
    ab, ac = (b - a)[None], (c - a)[None]
    ap, bp, cp = p - a[None], p - b[None], p - c[None]
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    tiny = torch.finfo(p.dtype).tiny

    # Inside: barycentric coordinates (the default; the region tests below replace it where they apply).
    denom = (va + vb + vc).clamp(min=tiny)
    v, w = vb / denom, vc / denom
    closest = a[None] + ab * v[..., None] + ac * w[..., None]

    def put(mask, point):
        nonlocal closest
        closest = torch.where(mask[..., None], point, closest)

    # Ericson tests in this order: corner a, corner b, edge ab, corner c, edge ac, edge bc, inside; the
    # first test that applies wins. Here the tests are applied in the reverse order, each one over the
    # result of the ones before, so that the earlier test of Ericson's order has the last word.
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=tiny)
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b[None] + (c - b)[None] * t[..., None])  # edge bc
    t = d2 / (d2 - d6).clamp(min=tiny)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a[None] + ac * t[..., None])  # edge ac
    put((d6 >= 0) & (d5 <= d6), c[None].expand_as(closest))  # corner c
    t = d1 / (d1 - d3).clamp(min=tiny)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a[None] + ab * t[..., None])  # edge ab
    put((d3 >= 0) & (d4 <= d3), b[None].expand_as(closest))  # corner b
    put((d1 <= 0) & (d2 <= 0), a[None].expand_as(closest))  # corner a
    return ((p - closest) ** 2).sum(-1)


def point_to_surface(points: torch.Tensor, mesh: Meshes) -> torch.Tensor:
    """Squared distance of each point (P, 3) to the nearest triangle of a single mesh: (P,).

    Points and triangles are processed in blocks of about POINT_TRIANGLE_BLOCK pairs.
    """
    tris = mesh.verts_packed()[mesh.faces_packed()]  # (T, 3, 3)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    rows = max(1, POINT_TRIANGLE_BLOCK // max(1, len(tris)))
    cols = max(1, POINT_TRIANGLE_BLOCK // rows)
    best = torch.full((len(points),), float("inf"), device=points.device)
    for i in range(0, len(points), rows):
        for j in range(0, len(tris), cols):
            d = point_triangle_sq_distance(points[i:i + rows], a[j:j + cols], b[j:j + cols], c[j:j + cols])
            best[i:i + rows] = torch.minimum(best[i:i + rows], d.min(dim=1).values)
    return best


def distance_bound(points_a: torch.Tensor, points_b: torch.Tensor, dense_a: torch.Tensor, dense_b: torch.Tensor) -> float:
    """A fast upper bound of `surface_distance`: the same measure with the nearest dense sample point in place
    of the nearest surface point (a sample point lies on the surface, so it is never nearer than the surface).

    Args:
        points_a, points_b: The measured points of each mesh (P, 3).
        dense_a, dense_b: Dense surface samples of each mesh (D, 3), for example BOUND_POINTS points.

    The bound is at most about half the spacing of the dense sample above the exact value (about 0.005
    with 50,000 points), so a bound above the twin range proves that the pair is not a twin.
    """
    ab = knn_points(points_a[None], dense_b[None], K=1).dists[0, :, 0].mean()
    ba = knn_points(points_b[None], dense_a[None], K=1).dists[0, :, 0].mean()
    return float(0.5 * (ab + ba)) ** 0.5


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

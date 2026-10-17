"""Surface normals for shading, with sharp edges kept sharp ("crease angle" smoothing).

A normal is the direction that a surface faces. The renderer needs one at every pixel. Each
triangle has three corners, and the normal at a pixel is blended from the normals at the corners
of its triangle. The corner normals decide how the surface looks:

- flat: every corner gets the normal of its own triangle. Round surfaces show their triangles.
- smooth: every corner gets the average normal of all triangles that share its vertex. Round
  surfaces look round, but at a sharp edge (for example the 90-degree edge of a box) the average
  mixes the two sides, and flat panels get false gradients.
- crease (the project default): a corner averages only the neighbouring triangles that meet its
  own triangle at less than the crease angle (for example 30 degrees). Round surfaces look round,
  and sharp edges stay sharp. 3D programs call this "auto smooth".

Many meshes have triangles whose stored front side points the wrong way ("inconsistent winding"),
so a neighbour's normal can point almost opposite to the triangle's own normal. The crease
comparison therefore uses the neighbour's normal turned to the side of the triangle's normal: a
flipped neighbour still counts as a smooth neighbour. (The renderer later turns every normal
towards the camera.)
"""

import math

import torch

NORMAL_MODES = ("flat", "smooth", "crease")
MAX_CREASE_PAIRS = 60_000_000  # above this many triangle pairs, "crease" falls back to "smooth" (GPU memory)


def face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the unit normal and the area of every triangle: (F, 3) and (F,).

    The normal follows the right-hand rule over the corners (a, b, c): (b - a) x (c - a).
    The length of that cross product is twice the triangle area.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    cross = torch.linalg.cross(b - a, c - a)
    doubled_area = cross.norm(dim=1)
    return cross / doubled_area.clamp_min(1e-20)[:, None], doubled_area / 2


def corner_normals(vertices: torch.Tensor, faces: torch.Tensor, mode: str = "crease", crease_deg: float = 30.0) -> torch.Tensor:
    """Return one normal for each corner of each triangle: (F, 3 corners, 3).

    Args:
        vertices: float tensor (V, 3).
        faces: int64 tensor (F, 3).
        mode: "flat", "smooth", or "crease" (see the module description).
        crease_deg: In "crease" mode, neighbouring triangles that meet at more than this angle
            are not blended.

    "crease" compares every pair of triangles that share a vertex: about 18 pairs per triangle in
    real meshes (8.5 million for a 480,000-triangle ABO mesh). For a mesh with more than
    MAX_CREASE_PAIRS pairs, "smooth" is used instead.
    """
    if mode not in NORMAL_MODES:
        raise ValueError(f"mode must be one of {NORMAL_MODES}, got {mode!r}")
    normals, areas = face_normals(vertices, faces)
    if mode == "flat":
        return normals[:, None, :].expand(-1, 3, -1).contiguous()

    weighted = normals * areas[:, None]  # larger triangles count more in an average
    if mode == "crease":
        degree = torch.bincount(faces.reshape(-1), minlength=len(vertices))
        if int((degree.long() ** 2).sum()) > MAX_CREASE_PAIRS:
            mode = "smooth"
    if mode == "smooth":
        vertex_sum = torch.zeros_like(vertices).index_add_(0, faces.reshape(-1), weighted.repeat_interleave(3, dim=0))
        return torch.nn.functional.normalize(vertex_sum[faces], dim=-1)

    # mode == "crease". Each (triangle, corner) is an "incidence" of a vertex. For every pair of
    # incidences of the same vertex, decide whether the two triangles are blended.
    face_of = torch.arange(len(faces), device=faces.device).repeat_interleave(3)  # (3F,) triangle of each incidence
    vertex_of = faces.reshape(-1)  # (3F,) vertex of each incidence
    order = torch.argsort(vertex_of)
    vertex_sorted, face_sorted = vertex_of[order], face_of[order]
    group_size = torch.bincount(vertex_sorted, minlength=len(vertices))  # triangles around each vertex
    group_start = torch.cumsum(group_size, 0) - group_size
    size_per_incidence = group_size[vertex_sorted]
    # All pairs (i, j) of incidences that share a vertex: i repeated once per member of its group.
    first = torch.arange(len(vertex_sorted), device=faces.device).repeat_interleave(size_per_incidence)
    offset_in_group = torch.arange(len(first), device=faces.device) - torch.repeat_interleave(
        torch.cumsum(size_per_incidence, 0) - size_per_incidence, size_per_incidence
    )
    second = group_start[vertex_sorted[first]] + offset_in_group
    own, neighbour = face_sorted[first], face_sorted[second]

    cosine = (normals[own] * normals[neighbour]).sum(dim=1)
    side = torch.where(cosine < 0, -1.0, 1.0).to(vertices.dtype)  # turn a flipped neighbour to the own side
    blended = cosine.abs() >= math.cos(math.radians(crease_deg))  # a triangle is always blended with itself
    contribution = weighted[neighbour] * (side * blended)[:, None]
    sums = torch.zeros(len(vertex_sorted), 3, dtype=vertices.dtype, device=vertices.device).index_add_(0, first, contribution)

    result = torch.empty(len(faces) * 3, 3, dtype=vertices.dtype, device=vertices.device)
    result[order] = torch.nn.functional.normalize(sums, dim=-1)  # back from vertex order to (triangle, corner) order
    return result.reshape(len(faces), 3, 3)

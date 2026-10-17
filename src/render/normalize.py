"""Move and scale a mesh so that it fits exactly in the unit sphere.

Meshes come in different units and positions (Pix3D: various; ABO: metres, often standing on the
floor). Before rendering, every mesh gets the same position and size:

1. Find the centre of the bounding box (the box aligned with the x, y, z axes around all vertices).
2. Move the mesh so that this centre is at the origin.
3. Find the vertex farthest from the origin. Its distance is the radius.
4. Divide all vertex positions by the radius.

After this, the farthest vertex is at distance 1 from the origin, so the mesh fits in a sphere of
radius 1. A sphere has the same size from every direction, so one camera distance shows the whole
mesh from every camera (see `src/render/rig.py`).

Front direction: the rig expects the front of every mesh towards +Z (so that view 0, at azimuth 0,
shows the front). ABO meshes face +Z (the glTF convention). Pix3D meshes face -Z: most Pix3D photos
are taken from the -Z side of their mesh (7,922 of 10,069, from the camera poses in pix3d.json).
`turn_to_front` turns a mesh about the vertical axis so that its front faces +Z.

Why not scale the longest side of the box to 1 (the v1 method)? Then a corner of the box can be up
to 0.87 from the centre in one direction and only 0.5 in another, so the size in the image depends
on the camera direction.
"""

import torch


class DegenerateMeshError(ValueError):
    """All vertices are at the same point, so the mesh has no size to normalize."""


def normalize_to_unit_sphere(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Move the bounding-box centre to the origin and scale the farthest vertex to distance 1.

    Args:
        vertices: float tensor (V, 3).

    Returns:
        normalized: float tensor (V, 3); the farthest vertex is at distance 1.0.
        centre: tensor (3,), the original bounding-box centre.
        radius: the original distance of the farthest vertex from the centre.

    Raises:
        DegenerateMeshError: If all vertices are at the same point.
    """
    centre = (vertices.min(dim=0).values + vertices.max(dim=0).values) / 2
    centred = vertices - centre
    radius = float(centred.norm(dim=1).max())
    if radius == 0.0:
        raise DegenerateMeshError("all vertices are at the same point")
    return centred / radius, centre, radius


# Turns about the vertical (+Y) axis that bring each front direction to +Z: (x, y, z) -> new (x, y, z).
_TURNS = {
    "+z": lambda v: v,
    "-z": lambda v: v * v.new_tensor([-1.0, 1.0, -1.0]),  # 180 degrees
    "+x": lambda v: v[:, [2, 1, 0]] * v.new_tensor([-1.0, 1.0, 1.0]),  # 90 degrees: +x goes to +z
    "-x": lambda v: v[:, [2, 1, 0]] * v.new_tensor([1.0, 1.0, -1.0]),  # 90 degrees the other way: -x goes to +z
}
FRONT_AXES = tuple(_TURNS)


def turn_to_front(vertices: torch.Tensor, front_axis: str) -> torch.Tensor:
    """Turn a mesh about the vertical axis so that its front (`front_axis`, e.g. "-z") faces +Z.

    A turn about the vertical axis keeps +Y up and keeps the mesh a proper (not mirrored) copy.
    """
    if front_axis not in _TURNS:
        raise ValueError(f"front_axis must be one of {FRONT_AXES}, got {front_axis!r}")
    return _TURNS[front_axis](vertices)


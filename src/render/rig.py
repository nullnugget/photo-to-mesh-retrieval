"""The camera rig: where the cameras stand around a mesh, and how far away they are.

Coordinates: +Y is up and +Z is the front of the mesh (the Pix3D and ABO convention). After
normalization (`src/render/normalize.py`), the mesh fits in the unit sphere around the origin.
Every camera looks at the origin.

A camera position is given by two angles:

- elevation: the angle above the horizontal plane through the origin. 0 = level with the mesh
  centre, +90 = straight above, negative = below.
- azimuth: the angle around the vertical (+Y) axis. 0 = in front of the mesh (on the +Z axis);
  90 = on the right of the mesh as seen from the front (on the +X axis).

Two rigs are available (`render.rig` in `configs/render.yaml`):

- "rings25" (the project rig): 3 rings of 8 cameras at -15, +15, and +45 degrees elevation, plus
  1 camera from the top at +85 degrees. Most cameras are at the heights from which people take
  photos (furniture stands on the floor; photos are taken from standing height). The middle ring
  is turned by 22.5 degrees, so that its cameras sit between those of the other rings, like bricks
  in a wall. The top camera is at 85 degrees, not 90: looking exactly down, the direction "up"
  of the image would be undefined.
- "icosahedron12" (the v1 rig): one camera at each of the 12 corners of an icosahedron, evenly
  spread over the whole sphere. Four of them are below the mesh.

Camera distance: the mesh must be fully inside every image, from every direction. Seen from a
distance d, the unit sphere fills an angle of 2 * asin(1 / d). It fits in the field of view when
asin(1 / d) <= fov / 2, that is when d >= 1 / sin(fov / 2). A margin factor (1.05) keeps a small
border. With a field of view of 40 degrees: d = 1.05 / sin(20 degrees) = 3.07.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """The position of one camera of the rig (it always looks at the origin)."""

    index: int  # view number; the render of this camera is saved as view_<index>.png
    elevation_deg: float  # angle above the horizontal plane through the origin
    azimuth_deg: float  # angle around +Y; 0 = front (+Z), 90 = right (+X)

    def direction(self) -> np.ndarray:
        """Return the unit vector from the origin towards the camera."""
        elevation, azimuth = math.radians(self.elevation_deg), math.radians(self.azimuth_deg)
        return np.array(
            [
                math.cos(elevation) * math.sin(azimuth),  # x: right
                math.sin(elevation),  # y: up
                math.cos(elevation) * math.cos(azimuth),  # z: front
            ]
        )


def rings_rig(rings: list[dict], top: dict | None) -> list[CameraPose]:
    """Make a rig of camera rings, plus an optional camera from the top.

    Args:
        rings: One dict per ring, with "elevation_deg", "count" (cameras in the ring), and
            "azimuth_offset_deg" (the azimuth of the first camera; the others follow at equal steps).
        top: A dict with "elevation_deg" for the top camera, or None for no top camera.

    Returns:
        The cameras, ring by ring (in the order of `rings`), then the top camera.
    """
    poses = []
    for ring in rings:
        step = 360.0 / ring["count"]
        for i in range(ring["count"]):
            azimuth = (ring.get("azimuth_offset_deg", 0.0) + i * step) % 360.0
            poses.append(CameraPose(len(poses), float(ring["elevation_deg"]), azimuth))
    if top is not None:
        poses.append(CameraPose(len(poses), float(top["elevation_deg"]), 0.0))
    return poses


def icosahedron_rig() -> list[CameraPose]:
    """Make the 12-camera rig: one camera at each corner of an icosahedron.

    The 12 corners of an icosahedron are the cyclic permutations of (0, +-1, +-phi), where phi is
    the golden ratio. They are spread evenly over the sphere. (The order of the cameras can differ
    from the v1 code; the set of directions is the same kind of rig.)
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    corners = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            corners += [(0.0, a, b), (a, b, 0.0), (b, 0.0, a)]
    poses = []
    for x, y, z in corners:
        length = math.sqrt(x * x + y * y + z * z)
        elevation = math.degrees(math.asin(y / length))
        azimuth = math.degrees(math.atan2(x, z)) % 360.0
        poses.append(CameraPose(len(poses), elevation, azimuth))
    return poses


def camera_distance(fov_deg: float, margin: float = 1.05) -> float:
    """Return the distance at which the unit sphere fits in the field of view, times `margin`."""
    return margin / math.sin(math.radians(fov_deg) / 2.0)


def rig_from_config(render_cfg: dict) -> list[CameraPose]:
    """Make the rig that `render.rig` names ("rings25" or "icosahedron12")."""
    name = render_cfg["rig"]
    if name == "rings25":
        return rings_rig(render_cfg["rings"], render_cfg.get("top"))
    if name == "icosahedron12":
        return icosahedron_rig()
    raise ValueError(f"unknown rig {name!r}; choose rings25 or icosahedron12")

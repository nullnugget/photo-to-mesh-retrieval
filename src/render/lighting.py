"""Light setups and shading: how bright each pixel of a grey render is.

Every mesh has the same grey material, so the brightness of a surface comes only from its angle to
the lights. This brightness pattern ("shading") is the only cue to the 3D form in a grey render.

Shading model (Phong), for one pixel of the surface:

    brightness = albedo * (ambient + sum over lights of  diffuse_i * max(0, n . l_i))
                 + sum over lights of  specular_i * max(0, r_i . v) ** shininess

    n    unit surface normal at the pixel (the direction the surface faces)
    l_i  unit vector from the surface towards light i
    v    unit vector from the surface towards the camera
    r_i  the mirror direction of l_i about n: r_i = 2 (n . l_i) n - l_i

- ambient: light that comes from everywhere; it keeps surfaces that face away from all lights
  from going black.
- diffuse: matte reflection. A surface that faces the light (n . l = 1) gets the full value; a
  surface at 90 degrees to the light gets nothing. This is what shows the form.
- specular: the highlight, where the light is mirrored towards the camera. `shininess` sets its
  size (higher = smaller and sharper). Highlights are kept small, because they depend on the
  material, which the grey renders do not have.

Light types (the presets L0 to L4 in `configs/render.yaml`):

- "point_at_camera": one light at the camera position (v1). Then l = v for every pixel. Surfaces
  that face the camera are bright, and there are almost no shading differences ("flat" images).
- "directional_camera_relative": parallel light from a fixed direction relative to the camera,
  for example 35 degrees above and 35 degrees to the left of the viewing direction. It moves with
  the camera, so every view gets the same kind of shading.
- "key_and_fill": two camera-relative lights: a main ("key") light and a weaker "fill" light from
  the other side, which softens the dark side.
- "directional_world": parallel light from a fixed direction in the world (for example from
  straight above, like a ceiling light). It does not move with the camera.

The preset's specular weight is given to the first light only (the key light in "key_and_fill"),
so that a fill light adds no second highlight.
"""

import math
from dataclasses import dataclass

import torch

POINT_AT_CAMERA = "point_at_camera"
CAMERA_RELATIVE = "camera_relative"
WORLD_DIRECTION = "world_direction"


@dataclass(frozen=True)
class Light:
    """One light of a setup."""

    kind: str  # POINT_AT_CAMERA, CAMERA_RELATIVE, or WORLD_DIRECTION
    diffuse: float  # weight of the matte reflection
    specular: float  # weight of the highlight
    elevation_deg: float = 0.0  # CAMERA_RELATIVE: angle above the viewing direction
    azimuth_deg: float = 0.0  # CAMERA_RELATIVE: angle to the right of the viewing direction (negative = left)
    travel_direction: tuple[float, float, float] = (0.0, -1.0, 0.0)  # WORLD_DIRECTION: the way the light travels


@dataclass(frozen=True)
class LightSetup:
    """A complete light setup: the ambient light plus one or more lights."""

    name: str
    ambient: float
    lights: tuple[Light, ...]


def setup_from_preset(name: str, preset: dict) -> LightSetup:
    """Turn one entry of `render.lighting_presets` into a LightSetup.

    Raises:
        ValueError: If the preset type is unknown.
    """
    kind = preset["type"]
    ambient, specular = preset["ambient"], preset["specular"]
    if kind == "point_at_camera":
        lights = [Light(POINT_AT_CAMERA, preset["diffuse"], specular)]
    elif kind == "directional_camera_relative":
        lights = [Light(CAMERA_RELATIVE, preset["diffuse"], specular, preset["elevation_deg"], preset["azimuth_deg"])]
    elif kind == "key_and_fill":
        key, fill = preset["key"], preset["fill"]
        lights = [
            Light(CAMERA_RELATIVE, key["diffuse"], specular, key["elevation_deg"], key["azimuth_deg"]),
            Light(CAMERA_RELATIVE, fill["diffuse"], 0.0, fill["elevation_deg"], fill["azimuth_deg"]),  # no highlight
        ]
    elif kind == "directional_world":
        lights = [Light(WORLD_DIRECTION, preset["diffuse"], specular, travel_direction=tuple(preset["direction"]))]
    else:
        raise ValueError(f"unknown light type {kind!r} in preset {name}")
    return LightSetup(name, ambient, tuple(lights))


def setup_from_config(render_cfg: dict, name: str | None = None) -> LightSetup:
    """Return the light setup `name`, or the active one (`render.lighting`) if no name is given."""
    name = name or render_cfg["lighting"]
    return setup_from_preset(name, render_cfg["lighting_presets"][name])


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    """Scale vectors (last dimension = 3) to length 1. Zero vectors stay zero."""
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def towards_light(light: Light, camera_frame: dict[str, torch.Tensor], to_camera: torch.Tensor) -> torch.Tensor:
    """Return the unit vectors from the surface towards one light, for every pixel.

    Args:
        light: The light.
        camera_frame: Unit vectors of each view's camera in world coordinates, each (views, 3):
            "right", "up", and "back" (from the mesh towards the camera).
        to_camera: Unit vectors from each pixel towards its camera, (views, H, W, 3).

    Returns:
        Tensor (views, H, W, 3), or a tensor that broadcasts to it.
    """
    if light.kind == POINT_AT_CAMERA:
        return to_camera  # the light is at the camera, so it comes from the camera direction
    if light.kind == CAMERA_RELATIVE:
        # Start from the direction towards the camera ("back"), turn it to the right by the azimuth,
        # then raise it by the elevation. Negative azimuth = light from the left.
        elevation, azimuth = math.radians(light.elevation_deg), math.radians(light.azimuth_deg)
        direction = math.cos(elevation) * (
            math.sin(azimuth) * camera_frame["right"] + math.cos(azimuth) * camera_frame["back"]
        ) + math.sin(elevation) * camera_frame["up"]
        return _unit(direction)[:, None, None, :]  # the same direction for all pixels of a view
    if light.kind == WORLD_DIRECTION:
        travel = torch.tensor(light.travel_direction, dtype=to_camera.dtype, device=to_camera.device)
        return _unit(-travel).expand_as(to_camera)  # towards the light = against the way it travels
    raise ValueError(f"unknown light kind {light.kind!r}")


def shade(
    normals: torch.Tensor,
    to_camera: torch.Tensor,
    camera_frame: dict[str, torch.Tensor],
    setup: LightSetup,
    albedo: float,
    shininess: float,
) -> torch.Tensor:
    """Compute the brightness of every pixel with the Phong model (see the module description).

    Args:
        normals: Unit surface normals, (views, H, W, 3). They must face the camera (n . v >= 0);
            the renderer turns normals that face away (see `src/render/renderer.py`).
        to_camera: Unit vectors from each pixel towards its camera, (views, H, W, 3).
        camera_frame: See `towards_light`.
        setup: The light setup.
        albedo: Brightness of the grey material (0 to 1).
        shininess: Exponent of the highlight.

    Returns:
        Brightness, (views, H, W), clamped to 0..1.
    """
    diffuse_sum = torch.zeros(normals.shape[:-1], dtype=normals.dtype, device=normals.device)
    specular_sum = torch.zeros_like(diffuse_sum)
    for light in setup.lights:
        to_light = towards_light(light, camera_frame, to_camera)
        n_dot_l = (normals * to_light).sum(dim=-1)  # cosine of the angle between surface and light
        diffuse_sum += light.diffuse * n_dot_l.clamp_min(0.0)
        if light.specular > 0.0:
            mirrored = 2.0 * n_dot_l[..., None] * normals - to_light  # r = 2 (n . l) n - l
            highlight = (mirrored * to_camera).sum(dim=-1).clamp_min(0.0) ** shininess
            specular_sum += light.specular * highlight * (n_dot_l > 0)  # no highlight on the dark side
    return (albedo * (setup.ambient + diffuse_sum) + specular_sum).clamp(0.0, 1.0)

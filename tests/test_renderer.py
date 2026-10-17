"""Tests for src/render/lighting.py and src/render/renderer.py.

The renderer tests run on the GPU when one is available, otherwise on the CPU (slower).
"""

import math

import numpy as np
import pytest
import torch

from src.config import load_config
from src.render.lighting import Light, LightSetup, CAMERA_RELATIVE, POINT_AT_CAMERA, setup_from_config, shade
from src.render.renderer import MeshRenderer, RenderSettings
from src.render.normalize import normalize_to_unit_sphere
from src.render.rig import CameraPose

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME = {"right": torch.tensor([[1.0, 0.0, 0.0]]), "up": torch.tensor([[0.0, 1.0, 0.0]]), "back": torch.tensor([[0.0, 0.0, 1.0]])}


def _box(x0, x1, y0, y1, z0, z1):
    """Vertices and outward-wound faces of an axis-aligned box."""
    vertices = torch.tensor([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    faces = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return vertices, faces


def test_all_presets_parse():
    render_cfg = load_config(["render.yaml"])["render"]
    for name in render_cfg["lighting_presets"]:
        setup = setup_from_config(render_cfg, name)
        assert setup.name == name and len(setup.lights) >= 1


def test_shading_values():
    to_camera = torch.tensor([0.0, 0.0, 1.0]).expand(1, 1, 2, 3)
    normals = torch.tensor([[[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]])  # one pixel faces the camera, one faces right
    setup = LightSetup("test", ambient=0.2, lights=(Light(POINT_AT_CAMERA, diffuse=0.5, specular=0.0),))
    brightness = shade(normals, to_camera, FRAME, setup, albedo=0.8, shininess=10)
    assert torch.allclose(brightness, torch.tensor([[[0.8 * (0.2 + 0.5), 0.8 * 0.2]]]))  # full diffuse, then ambient only


def test_camera_relative_light_from_the_left():
    to_camera = torch.tensor([0.0, 0.0, 1.0]).expand(1, 1, 2, 3)
    normals = torch.tensor([[[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])  # facing left, facing right
    setup = LightSetup("test", 0.0, (Light(CAMERA_RELATIVE, 1.0, 0.0, elevation_deg=0.0, azimuth_deg=-90.0),))
    brightness = shade(normals, to_camera, FRAME, setup, albedo=1.0, shininess=10)
    assert math.isclose(float(brightness[0, 0, 0]), 1.0, abs_tol=1e-6) and float(brightness[0, 0, 1]) == 0.0


@pytest.fixture(scope="module")
def renderer():
    settings = RenderSettings(render_size=128, output_size=64, crop_margin=0.0, background_rgb=(128, 128, 128),
                              fov_deg=40.0, distance_margin=1.05, albedo=0.7, shininess=10.0)
    poses = [CameraPose(0, 0.0, 0.0), CameraPose(1, 0.0, 90.0), CameraPose(2, 85.0, 0.0)]
    setup = LightSetup("test", 0.3, (Light(CAMERA_RELATIVE, 0.65, 0.05, 35.0, -35.0),))
    return MeshRenderer(settings, poses, setup, DEVICE)


def test_render_size_background_and_no_clipping(renderer):
    vertices, faces = _box(-1, 1, -1, 1, -1, 1)
    out = renderer.render(normalize_to_unit_sphere(vertices)[0], faces)
    assert out.images.shape == (3, 64, 64, 3) and out.images.dtype == np.uint8
    assert not out.touches_border.any()  # the unit sphere fits in every view
    assert (out.object_fraction > 0.05).all()


def test_images_are_not_mirrored(renderer):
    # A tall box on the right (+X) and a short box on the left (-X), seen from the front (azimuth 0).
    tall, tall_faces = _box(0.2, 1.0, -1.0, 1.0, -0.3, 0.3)
    short, short_faces = _box(-1.0, -0.2, -1.0, -0.2, -0.3, 0.3)
    vertices = torch.cat([tall, short])
    faces = torch.cat([tall_faces, short_faces + len(tall)])
    out = renderer.render(normalize_to_unit_sphere(vertices)[0], faces)
    front = out.images[0]
    top_rows = (front[:16] != 128).any(axis=2)  # the top quarter: only the tall box reaches it
    columns = np.nonzero(top_rows.any(axis=0))[0]
    assert columns.min() > 32  # all of it is in the right half of the image

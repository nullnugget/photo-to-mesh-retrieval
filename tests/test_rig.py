"""Tests for src/render/rig.py."""

import math

import numpy as np
import pytest

from src.config import load_config
from src.render.rig import CameraPose, camera_distance, icosahedron_rig, rig_from_config


@pytest.fixture(scope="module")
def render_cfg():
    return load_config(["render.yaml"])["render"]


def test_rings25_has_the_agreed_cameras(render_cfg):
    poses = rig_from_config(render_cfg)
    assert len(poses) == 25 and [p.index for p in poses] == list(range(25))
    for ring, elevation, first_azimuth in [(range(0, 8), -15, 0.0), (range(8, 16), 15, 22.5), (range(16, 24), 45, 0.0)]:
        assert all(poses[i].elevation_deg == elevation for i in ring)
        assert [poses[i].azimuth_deg for i in ring] == [first_azimuth + 45.0 * k for k in range(8)]
    assert poses[24].elevation_deg == 85


def test_directions():
    assert np.allclose(CameraPose(0, 0, 0).direction(), [0, 0, 1])  # azimuth 0 = front (+Z)
    assert np.allclose(CameraPose(0, 0, 90).direction(), [1, 0, 0])  # azimuth 90 = right (+X)
    assert np.allclose(CameraPose(0, 90, 0).direction(), [0, 1, 0], atol=1e-12)  # elevation 90 = above


def test_unit_sphere_fits_in_every_view(render_cfg):
    distance = camera_distance(render_cfg["fov_deg"], render_cfg["distance_margin"])
    half_angle_of_sphere = math.degrees(math.asin(1.0 / distance))
    assert half_angle_of_sphere < render_cfg["fov_deg"] / 2
    assert math.isclose(camera_distance(40.0, 1.05), 3.07, abs_tol=0.01)


def test_icosahedron_rig_is_even():
    directions = np.stack([p.direction() for p in icosahedron_rig()])
    assert len(directions) == 12
    assert np.allclose(directions.sum(axis=0), 0.0, atol=1e-9)  # evenly spread: the directions cancel out
    assert sum(d[1] < -0.1 for d in directions) == 4  # four cameras below the mesh, as in v1

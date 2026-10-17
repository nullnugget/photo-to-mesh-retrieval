"""Tests for src/data/glb.py, on small GLB files built in memory (no download needed)."""

import json
import struct

import numpy as np
import pytest

from src.data import glb

CUBE = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float32)  # 8 corners
TRIANGLES = np.array([[0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7]], dtype=np.uint16)


def _glb(gltf: dict, binary: bytes) -> bytes:
    """Build GLB bytes from a glTF document and its binary chunk (both padded to 4 bytes)."""
    json_bytes = json.dumps(gltf).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), glb.JSON_CHUNK_TYPE) + json_bytes
        + struct.pack("<II", len(binary), glb.BIN_CHUNK_TYPE) + binary
    )


def _cube_file(node: dict, interleaved: bool = False, padding: int = 0) -> tuple[bytes, dict]:
    """A cube mesh with 4 triangles. `padding` bytes of other data come first (like a texture image)."""
    pad = b"\x7f" * padding
    if interleaved:  # position and a dummy normal per vertex, in one bufferView with a stride of 24 bytes
        vertex_data = np.concatenate([CUBE, np.ones_like(CUBE)], axis=1).astype("<f4").tobytes()
        views = [{"buffer": 0, "byteOffset": padding, "byteLength": len(vertex_data), "byteStride": 24}]
    else:
        vertex_data = CUBE.astype("<f4").tobytes()
        views = [{"buffer": 0, "byteOffset": padding, "byteLength": len(vertex_data)}]
    index_data = TRIANGLES.astype("<u2").tobytes()
    views.append({"buffer": 0, "byteOffset": padding + len(vertex_data), "byteLength": len(index_data)})
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [node],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 12, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": padding + len(vertex_data) + len(index_data)}],
    }
    return _glb(gltf, pad + vertex_data + index_data), gltf


def test_header_and_full_file_decoding(tmp_path):
    raw, _ = _cube_file({"mesh": 0})
    total, json_length = glb.parse_header(raw[:20])
    assert total == len(raw) and json_length % 4 == 0
    (tmp_path / "cube.glb").write_bytes(raw)
    _, vertices, faces = glb.read_glb_file(tmp_path / "cube.glb")
    assert np.allclose(vertices, CUBE) and np.array_equal(faces, TRIANGLES.astype(np.int64))


def test_node_placement_is_applied():
    # translate by (10, 0, 0), scale by 2, rotate 90 degrees about +Y (quaternion x, y, z, w)
    s = np.sqrt(0.5)
    raw, gltf = _cube_file({"mesh": 0, "translation": [10, 0, 0], "scale": [2, 2, 2], "rotation": [0, s, 0, s]})
    _, json_length = glb.parse_header(raw[:20])
    vertices, _ = glb.decode_geometry(gltf, raw[glb.bin_chunk_start(json_length) :])
    corner = vertices[1]  # the corner (0, 0, 1): rotated to (1, 0, 0), scaled to (2, 0, 0), moved to (12, 0, 0)
    assert np.allclose(corner, [12, 0, 0], atol=1e-6)


def test_parent_matrix_is_applied():
    raw, gltf = _cube_file({"mesh": 0})
    gltf["nodes"] = [{"children": [1], "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1]}, {"mesh": 0}]  # parent: +5 in y
    gltf["scenes"] = [{"nodes": [0]}]
    _, json_length = glb.parse_header(raw[:20])
    vertices, _ = glb.decode_geometry(gltf, raw[glb.bin_chunk_start(json_length) :])
    assert np.allclose(vertices, CUBE + [0, 5, 0])


def test_byte_range_skips_other_data_and_decodes_alone():
    raw, gltf = _cube_file({"mesh": 0}, interleaved=True, padding=1000)
    start, end = glb.geometry_byte_range(gltf)
    assert start == 1000  # the 1,000 bytes before the geometry are not part of the range
    _, json_length = glb.parse_header(raw[:20])
    bin_start = glb.bin_chunk_start(json_length)
    vertices, faces = glb.decode_geometry(gltf, raw[bin_start + start : bin_start + end], data_start=start)
    assert np.allclose(vertices, CUBE) and len(faces) == 4


def test_compressed_files_are_rejected():
    _, gltf = _cube_file({"mesh": 0})
    gltf["extensionsRequired"] = ["KHR_draco_mesh_compression"]
    with pytest.raises(glb.UnsupportedGlb):
        glb.geometry_byte_range(gltf)
    with pytest.raises(glb.UnsupportedGlb):
        glb.parse_header(b"not a glb file at all")


def test_ply_round_trip_with_pytorch3d(tmp_path):
    from pytorch3d.io import load_ply

    path = glb.write_ply(tmp_path / "cube.ply", CUBE, TRIANGLES.astype(np.int64))
    vertices, faces = load_ply(path)
    assert np.allclose(vertices.numpy(), CUBE) and np.array_equal(faces.numpy(), TRIANGLES.astype(np.int64))
    assert not (tmp_path / "cube.ply.part").exists()

"""Read only the geometry of a binary glTF (`.glb`) file, and write it as a PLY mesh.

ABO meshes are `.glb` files of 1 to 100 MB. Most of each file is texture images; the geometry
(vertex positions and triangle indices) is about 5 % of it. The renderer uses a grey material and
no textures, so only the geometry is downloaded. This module gives the pieces for that:

1. `parse_header`: the first 20 bytes of a GLB file give the total length and the length of the
   JSON chunk.
2. The JSON chunk (the "glTF document") describes the file. Its parts used here:
   - "accessors": typed arrays, for example "3 float32 values per vertex, 10,990 vertices";
   - "bufferViews": byte ranges of the binary chunk, where the accessors keep their data;
   - "meshes": each mesh has "primitives"; each primitive names the accessor of its vertex
     positions ("POSITION") and of its triangle indices ("indices");
   - "nodes" and "scenes": where each mesh is placed (translation, rotation, scale, or a 4 x 4
     matrix, and a parent-child tree).
3. `geometry_byte_range`: the smallest byte range of the binary chunk that contains all
   positions and indices. One HTTP range request fetches it.
4. `decode_geometry`: reads the arrays from that range, applies the placement of each node, and
   joins all triangle primitives into one vertex array and one face array.

`write_ply` saves the result as a binary PLY file, which PyTorch3D reads directly
(`pytorch3d.io.load_ply`). The JSON chunk is kept next to it, so that each PLY file can be traced
back to its source file.

GLB layout (all numbers little-endian):

    bytes 0-3    magic "glTF"         bytes 4-7    version (2)       bytes 8-11  total length
    bytes 12-15  JSON chunk length    bytes 16-19  chunk type "JSON"
    JSON chunk (padded with spaces to a multiple of 4 bytes)
    8 bytes: BIN chunk length and chunk type "BIN\\0"
    BIN chunk: all binary data; bufferView byte offsets count from its first byte
"""

import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = b"glTF"
JSON_CHUNK_TYPE = 0x4E4F534A  # "JSON" read as a little-endian uint32
BIN_CHUNK_TYPE = 0x004E4942  # "BIN\0"
HEADER_BYTES = 20  # 12-byte file header + 8-byte header of the JSON chunk
TRIANGLES = 4  # glTF primitive mode for triangle lists (the default)

# glTF accessor component types -> numpy dtypes, and accessor types -> values per element.
COMPONENT_DTYPES = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class UnsupportedGlb(ValueError):
    """The file uses a feature that this reader does not handle (for example Draco compression)."""


def parse_header(first_bytes: bytes) -> tuple[int, int]:
    """Read the first 20 bytes of a GLB file.

    Returns:
        (total_length, json_length): the file length in bytes, and the JSON chunk length.

    Raises:
        UnsupportedGlb: If the bytes are not the start of a GLB 2 file.
    """
    if len(first_bytes) < HEADER_BYTES:
        raise UnsupportedGlb(f"need {HEADER_BYTES} header bytes, got {len(first_bytes)}")
    magic, version, total_length, json_length, chunk_type = struct.unpack("<4sIIII", first_bytes[:HEADER_BYTES])
    if magic != GLB_MAGIC or version != 2 or chunk_type != JSON_CHUNK_TYPE:
        raise UnsupportedGlb(f"not a GLB 2 file (magic {magic!r}, version {version})")
    return total_length, json_length


def bin_chunk_start(json_length: int) -> int:
    """Return the file offset of the first byte of the BIN chunk data."""
    return HEADER_BYTES + json_length + 8  # + the 8-byte header of the BIN chunk


def _check_supported(gltf: dict) -> None:
    """Raise UnsupportedGlb for compressed or external geometry."""
    required = gltf.get("extensionsRequired") or []
    if required:
        raise UnsupportedGlb(f"required extensions {required} (for example Draco compression)")
    for view in gltf.get("bufferViews", []):
        if view.get("buffer", 0) != 0:
            raise UnsupportedGlb("geometry in an external buffer")


def _node_matrix(node: dict) -> np.ndarray:
    """Return the 4 x 4 local placement matrix of one node.

    A node gives either "matrix" (16 values, column by column) or translation, rotation (a
    quaternion x, y, z, w), and scale. The matrix is translation x rotation x scale.
    """
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF stores columns first
    tx, ty, tz = node.get("translation", [0.0, 0.0, 0.0])
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    sx, sy, sz = node.get("scale", [1.0, 1.0, 1.0])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array([sx, sy, sz])  # scales the columns: rotation x scale
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def mesh_placements(gltf: dict) -> list[tuple[int, np.ndarray]]:
    """Return (mesh index, 4 x 4 world matrix) for every node of the scene that shows a mesh.

    The world matrix of a node is the product of the local matrices from the root to the node.
    A file without scenes shows each mesh once, unmoved.
    """
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    if not scenes:
        return [(i, np.eye(4)) for i in range(len(gltf.get("meshes", [])))]
    placements = []
    stack = [(root, np.eye(4)) for root in scenes[gltf.get("scene", 0)].get("nodes", [])]
    while stack:  # walk the node tree from the roots
        index, parent_matrix = stack.pop()
        node = nodes[index]
        world = parent_matrix @ _node_matrix(node)
        if "mesh" in node:
            placements.append((node["mesh"], world))
        stack.extend((child, world) for child in node.get("children", []))
    return placements


def _triangle_primitives(gltf: dict) -> list[dict]:
    """Return every triangle primitive of every mesh that the scene uses (other modes are skipped)."""
    used = {mesh_index for mesh_index, _ in mesh_placements(gltf)}
    return [
        p
        for i, mesh in enumerate(gltf.get("meshes", []))
        if i in used
        for p in mesh["primitives"]
        if p.get("mode", TRIANGLES) == TRIANGLES
    ]


def _accessor_range(gltf: dict, accessor_index: int) -> tuple[int, int]:
    """Return the byte range [start, end) of the BIN chunk that holds the bufferView of one accessor."""
    accessor = gltf["accessors"][accessor_index]
    if "sparse" in accessor or "bufferView" not in accessor:
        raise UnsupportedGlb("sparse accessor or accessor without data")
    view = gltf["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0)
    return start, start + view["byteLength"]


def geometry_byte_range(gltf: dict) -> tuple[int, int]:
    """Return the smallest range [start, end) of the BIN chunk that holds all positions and indices.

    The range can contain other data between the parts (for example the vertex normals); that
    data is downloaded too but not used.

    Raises:
        UnsupportedGlb: For compressed geometry, or if no triangle primitive exists.
    """
    _check_supported(gltf)
    ranges = []
    for primitive in _triangle_primitives(gltf):
        ranges.append(_accessor_range(gltf, primitive["attributes"]["POSITION"]))
        if "indices" in primitive:
            ranges.append(_accessor_range(gltf, primitive["indices"]))
    if not ranges:
        raise UnsupportedGlb("no triangle geometry")
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def _read_accessor(gltf: dict, accessor_index: int, data: bytes, data_start: int) -> np.ndarray:
    """Read one accessor as an array of shape (count, values per element).

    Args:
        data: Bytes of the BIN chunk from offset `data_start` on (a part of the chunk is enough,
            if it contains the accessor).
    """
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[accessor["componentType"]])
    width = TYPE_SIZES[accessor["type"]]
    stride = view.get("byteStride") or dtype.itemsize * width  # byteStride > element size = interleaved data
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0) - data_start
    count = accessor["count"]
    if offset < 0 or offset + stride * (count - 1) + dtype.itemsize * width > len(data):
        raise UnsupportedGlb(f"accessor {accessor_index} is outside the downloaded byte range")
    return np.ndarray((count, width), dtype=dtype, buffer=data, offset=offset, strides=(stride, dtype.itemsize))


def decode_geometry(gltf: dict, data: bytes, data_start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return the placed vertices and the triangle faces of the whole scene, joined into one mesh.

    Args:
        gltf: The parsed JSON chunk.
        data: Bytes of the BIN chunk, from offset `data_start` on (for example the range that
            `geometry_byte_range` gave).
        data_start: Offset of `data[0]` in the BIN chunk.

    Returns:
        vertices: float32 array (V, 3), in the units of the file (metres for ABO).
        faces: int64 array (F, 3), indices into `vertices`.
    """
    _check_supported(gltf)
    all_vertices, all_faces, offset = [], [], 0
    for mesh_index, world in mesh_placements(gltf):
        for primitive in gltf["meshes"][mesh_index]["primitives"]:
            if primitive.get("mode", TRIANGLES) != TRIANGLES:
                continue
            positions = _read_accessor(gltf, primitive["attributes"]["POSITION"], data, data_start).astype(np.float64)
            placed = positions @ world[:3, :3].T + world[:3, 3]  # apply the node placement to each vertex
            if "indices" in primitive:
                faces = _read_accessor(gltf, primitive["indices"], data, data_start).astype(np.int64).reshape(-1, 3)
            else:  # no index list: every 3 vertices form one triangle
                faces = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)
            all_vertices.append(placed.astype(np.float32))
            all_faces.append(faces + offset)
            offset += len(positions)
    if not all_vertices:
        raise UnsupportedGlb("no triangle geometry")
    return np.concatenate(all_vertices), np.concatenate(all_faces)


def read_glb_file(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Read a complete local `.glb` file: (glTF JSON, vertices, faces). Used by tests and checks."""
    raw = Path(path).read_bytes()
    _, json_length = parse_header(raw[:HEADER_BYTES])
    gltf = json.loads(raw[HEADER_BYTES : HEADER_BYTES + json_length])
    vertices, faces = decode_geometry(gltf, raw[bin_chunk_start(json_length) :])
    return gltf, vertices, faces


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Write a binary little-endian PLY file with float32 vertices and int32 triangle faces.

    The file is written to a temporary name first and then renamed, so that an interrupted
    download never leaves a half-written PLY file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    face_records = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    face_records["count"] = 3
    face_records["indices"] = faces
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
        f.write(face_records.tobytes())
    temporary.replace(path)
    return path

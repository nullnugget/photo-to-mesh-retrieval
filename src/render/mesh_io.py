"""Load a mesh file as two arrays: vertex positions and triangle faces.

The renderer needs only the geometry, because every mesh gets the same grey material. Thus
textures, materials, and stored normals are not read.

Supported files:

- `.obj` (Pix3D). Faces with more than 3 corners are split into triangles by PyTorch3D.
- `.ply` (ABO, written by the download step).
"""

from pathlib import Path

import torch
from pytorch3d.io import load_obj, load_ply

from src import paths

MESH_SUFFIXES = (".obj", ".ply")


class EmptyMeshError(ValueError):
    """The file contains no vertices or no triangles."""


def load_mesh(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the geometry of one mesh file.

    Args:
        path: Mesh file, relative to the repository root (or absolute).

    Returns:
        vertices: float32 tensor (V, 3), in the units of the file.
        faces: int64 tensor (F, 3), indices into `vertices`.

    Raises:
        ValueError: If the file type is not supported.
        EmptyMeshError: If the file has no vertices or no faces.
    """
    path = paths.resolve(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        # load_textures=False: skip the material and texture files (not needed, and often missing).
        vertices, faces, _ = load_obj(path, load_textures=False)
        faces = faces.verts_idx  # load_obj also returns texture and normal indices; only the vertex indices are used
    elif suffix == ".ply":
        vertices, faces = load_ply(path)
    else:
        raise ValueError(f"unsupported mesh file {path.name}; supported: {MESH_SUFFIXES}")
    if len(vertices) == 0 or len(faces) == 0:
        raise EmptyMeshError(f"{path}: {len(vertices)} vertices, {len(faces)} faces")
    return vertices.float(), faces.long()

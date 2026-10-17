"""Make very large meshes smaller before rendering, with no visible change ("vertex clustering").

A few ABO meshes have millions of triangles (the largest has 11.5 million). Most of those triangles
are much smaller than one pixel of the render, so they do not change the image, but they need
more GPU memory than the 6 GB card has.

Method:

1. Put a 3D grid of small cubes ("cells") over the mesh. The cell size is about one pixel of the
   raw render (the unit sphere is about 490 pixels wide in the 512-pixel render, so one pixel is
   about 2 / 490 = 0.004 units).
2. Merge all vertices in the same cell into one vertex, at their average position.
3. Point every triangle to the merged vertices. A triangle with two or three corners in the same
   cell has collapsed to a line or a point; it is removed.

The surface keeps its shape to within one cell, so the render looks the same. Only meshes with
more triangles than a limit are simplified (`render.simplify_above_faces`); all others are
rendered exactly as they are.
"""

import torch


def merge_vertices_in_grid(vertices: torch.Tensor, faces: torch.Tensor, cell_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge the vertices of each grid cell and drop the collapsed triangles.

    Args:
        vertices: float tensor (V, 3), normalized to the unit sphere.
        faces: int64 tensor (F, 3).
        cell_size: Edge length of a grid cell, in the units of `vertices`.

    Returns:
        (vertices, faces) of the simplified mesh.
    """
    cells = torch.floor(vertices / cell_size).long()  # integer cell coordinates of each vertex
    _, merged_index = torch.unique(cells, dim=0, return_inverse=True)  # the same number for vertices in one cell
    count = torch.bincount(merged_index).to(vertices.dtype)
    merged = torch.zeros(len(count), 3, dtype=vertices.dtype, device=vertices.device).index_add_(0, merged_index, vertices)
    merged = merged / count[:, None]  # average position of the vertices in each cell
    new_faces = merged_index[faces]
    a, b, c = new_faces.unbind(dim=1)
    keep = (a != b) & (b != c) & (a != c)  # a triangle with two corners in one cell has collapsed
    return merged, new_faces[keep]

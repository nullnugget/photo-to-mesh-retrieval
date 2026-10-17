"""Render one mesh from every camera of the rig, as grey images framed like the query images.

Steps for one mesh (all views of a batch are rendered together on the GPU):

0. Simplify, only for very large meshes (more than `simplify_above_faces` triangles): their tiny
   triangles are invisible at this resolution but need too much GPU memory (`src/render/simplify.py`).
1. Rasterize: for every pixel of every view, find the triangle that is visible there, and where
   on the triangle the pixel falls (PyTorch3D `MeshRasterizer`, one triangle per pixel, no blur).
   Pixels without a triangle are background. This also gives the object mask (the "alpha").
2. Normals: interpolate a surface normal for every pixel from the normals at the corners of its
   triangle (`src/render/normals.py`; "crease" keeps sharp edges sharp and round surfaces round).
3. Two-sided surfaces: many meshes have triangles whose stored front side points inwards. A
   normal that points away from the camera is turned around, so that such a triangle is lit like
   its neighbours instead of going dark.
4. Shade: brightness from the light setup (`src/render/lighting.py`); background pixels get the
   background grey.
5. Crop: the same `crop_to_square` as the queries, with the mask from step 1: grey background,
   tight square crop, resize to 224 x 224. Thus renders and queries have the same framing.

The raw render is 512 x 512 pixels. The object is about 260 to 460 pixels across in it, so the
final 224-pixel image is still a reduction (no loss of detail from the camera distance).

Clipping check: the camera distance keeps the unit sphere inside every image, so no object pixel
should touch the image border. `RenderedViews.touches_border` reports it for every view.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings, look_at_view_transform
from pytorch3d.structures import Meshes

from src.preprocess.crop import crop_to_square
from src.render.lighting import LightSetup, shade, setup_from_config
from src.render.mesh_io import load_mesh
from src.render.normalize import normalize_to_unit_sphere, turn_to_front
from src.render.normals import NORMAL_MODES, corner_normals
from src.render.rig import CameraPose, camera_distance, rig_from_config
from src.render.simplify import merge_vertices_in_grid


@dataclass(frozen=True)
class RenderSettings:
    """The image and material settings of the renderer (from `render` in `configs/render.yaml`)."""

    render_size: int  # side length of the raw render, in pixels
    output_size: int  # side length after the crop
    crop_margin: float  # border around the object in the crop (as for the queries)
    background_rgb: tuple[int, int, int]  # background grey (as for the queries)
    fov_deg: float  # field of view of every camera
    distance_margin: float  # camera distance = distance_margin / sin(fov / 2)
    albedo: float  # brightness of the grey material, 0 to 1
    shininess: float  # size of the highlights (higher = smaller)
    normals: str = "crease"  # "flat", "smooth", or "crease" (see src/render/normals.py)
    crease_deg: float = 30.0  # in "crease" mode: triangles that meet at a larger angle are not blended
    max_faces_per_batch: int = 2_000_000  # GPU memory budget: views per batch = this / faces of the mesh (1 to 25)
    simplify_above_faces: int = 1_000_000  # meshes with more triangles are simplified first (src/render/simplify.py)
    simplify_cell: float = 0.004  # grid cell size of the simplification, about one pixel of the raw render

    @classmethod
    def from_config(cls, render_cfg: dict) -> "RenderSettings":
        """Read the settings from the `render` section of the configuration."""
        material = render_cfg["material"]
        return cls(
            render_size=render_cfg["render_size"],
            output_size=render_cfg["output_size"],
            crop_margin=render_cfg["crop_margin"],
            background_rgb=tuple(render_cfg["background_rgb"]),
            fov_deg=render_cfg["fov_deg"],
            distance_margin=render_cfg["distance_margin"],
            albedo=material["albedo"],
            shininess=material["shininess"],
            normals=material.get("normals", "crease"),
            crease_deg=material.get("crease_deg", 30.0),
            max_faces_per_batch=render_cfg.get("max_faces_per_batch", 2_000_000),
            simplify_above_faces=render_cfg.get("simplify_above_faces", 1_000_000),
            simplify_cell=render_cfg.get("simplify_cell", 0.004),
        )


@dataclass
class RenderedViews:
    """The renders of one mesh, one entry per camera of the rig."""

    images: np.ndarray  # uint8 (views, output_size, output_size, 3): cropped grey renders
    touches_border: np.ndarray  # bool (views,): True if the object touched the border of the raw render
    object_fraction: np.ndarray  # float (views,): share of the raw render covered by the object


def camera_frames(poses: list[CameraPose], device: torch.device) -> dict[str, torch.Tensor]:
    """Return the unit vectors "right", "up", and "back" of each camera in world coordinates, (views, 3).

    "back" points from the origin towards the camera. "right" and "up" are the directions of the
    image's x axis and y axis in the world (world up is +Y; the top camera is at 85 degrees, not 90,
    so "right" is always defined).
    """
    back = torch.tensor(np.stack([p.direction() for p in poses]), dtype=torch.float32, device=device)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(back)
    right = torch.nn.functional.normalize(torch.linalg.cross(world_up, back), dim=-1)
    up = torch.linalg.cross(back, right)
    return {"right": right, "up": up, "back": back}


class MeshRenderer:
    """Renders meshes with a fixed rig, light setup, and settings. Make one, then call `render` per mesh."""

    def __init__(self, settings: RenderSettings, poses: list[CameraPose], light_setup: LightSetup, device: str | torch.device = "cuda"):
        if settings.normals not in NORMAL_MODES:
            raise ValueError(f"normals must be one of {NORMAL_MODES}, got {settings.normals!r}")
        self.settings = settings
        self.poses = poses
        self.light_setup = light_setup
        self.device = torch.device(device)
        self.distance = camera_distance(settings.fov_deg, settings.distance_margin)

        # PyTorch3D cameras: each looks at the origin from its pose, with +Y as the up direction.
        # PyTorch3D uses the same angle convention as the rig: azimuth 0 = on the +Z axis.
        rotation, translation = look_at_view_transform(
            dist=self.distance,
            elev=[p.elevation_deg for p in poses],
            azim=[p.azimuth_deg for p in poses],
            device=self.device,
        )
        self.rotation, self.translation = rotation, translation
        self.frames = camera_frames(poses, self.device)
        self.camera_positions = self.frames["back"] * self.distance  # (views, 3)
        self.raster_settings = RasterizationSettings(
            image_size=settings.render_size,
            blur_radius=0.0,  # hard edges: a pixel is covered by a triangle or not
            faces_per_pixel=1,  # only the nearest triangle is needed
            cull_backfaces=False,  # keep triangles seen from behind (see step 3 in the module description)
        )

    @classmethod
    def from_config(cls, render_cfg: dict, device: str | torch.device = "cuda", lighting: str | None = None) -> "MeshRenderer":
        """Make a renderer from the `render` section; `lighting` overrides the active light setup."""
        return cls(RenderSettings.from_config(render_cfg), rig_from_config(render_cfg), setup_from_config(render_cfg, lighting), device)

    def _render_batch(self, vertices: torch.Tensor, faces: torch.Tensor, normals: torch.Tensor, views: slice) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize and shade some views. Returns (brightness (n, H, W), covered mask (n, H, W)).

        Args:
            normals: The corner normals of the mesh, (F, 3 corners, 3), from `corner_normals`.
        """
        count = len(range(*views.indices(len(self.poses))))
        cameras = FoVPerspectiveCameras(
            R=self.rotation[views], T=self.translation[views], fov=self.settings.fov_deg, device=self.device
        )
        meshes = Meshes(verts=[vertices], faces=[faces]).extend(count)  # the same mesh once per view
        fragments = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)(meshes)
        visible_face = fragments.pix_to_face  # (n, H, W, 1): index of the visible triangle, -1 = background
        covered = visible_face[..., 0] >= 0

        # Step 2: blend the corner values of the visible triangle at each pixel. The batch holds the
        # mesh once per view, so the triangle index of view k is k * F + f. The remainder f (the index
        # in the single mesh) selects the per-triangle values without copying them for every view.
        local_face = torch.where(visible_face >= 0, visible_face % len(faces), visible_face)
        pixel_normals = interpolate_face_attributes(local_face, fragments.bary_coords, normals)[..., 0, :]
        pixel_normals = torch.nn.functional.normalize(pixel_normals, dim=-1)
        positions = interpolate_face_attributes(local_face, fragments.bary_coords, vertices[faces])[..., 0, :]

        # Step 3: turn normals that point away from the camera.
        to_camera = torch.nn.functional.normalize(self.camera_positions[views][:, None, None, :] - positions, dim=-1)
        facing_away = (pixel_normals * to_camera).sum(dim=-1, keepdim=True) < 0
        pixel_normals = torch.where(facing_away, -pixel_normals, pixel_normals)

        # Step 4: shading.
        frames = {name: vectors[views] for name, vectors in self.frames.items()}
        brightness = shade(pixel_normals, to_camera, frames, self.light_setup, self.settings.albedo, self.settings.shininess)
        return brightness, covered

    def _crop_view(self, brightness: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Step 5 for one view: make the grey RGB image and crop it like a query.

        Args:
            brightness: uint8 (H, W), the shaded raw render.
            mask: bool (H, W), True where the object is.
        """
        background = np.asarray(self.settings.background_rgb, dtype=np.uint8)
        raw = np.where(mask[..., None], brightness[..., None].repeat(3, axis=2), background)
        return crop_to_square(raw, mask, self.settings.background_rgb, self.settings.crop_margin, self.settings.output_size)

    @torch.no_grad()
    def render(self, vertices: torch.Tensor, faces: torch.Tensor) -> RenderedViews:
        """Render a normalized mesh (inside the unit sphere) from every camera, and crop the views.

        Args:
            vertices: float tensor (V, 3), normalized with `normalize_to_unit_sphere`.
            faces: int64 tensor (F, 3).
        """
        vertices, faces = vertices.to(self.device, torch.float32), faces.to(self.device, torch.int64)
        if len(faces) > self.settings.simplify_above_faces:  # step 0, only for very large meshes
            vertices, faces = merge_vertices_in_grid(vertices, faces, self.settings.simplify_cell)
        normals = corner_normals(vertices, faces, self.settings.normals, self.settings.crease_deg)  # once per mesh
        images, touches, fractions = [], [], []
        # Views per batch: all 25 for a typical mesh, fewer for a large one (memory grows with views x faces).
        batch = max(1, min(len(self.poses), self.settings.max_faces_per_batch // len(faces)))
        start = 0
        while start < len(self.poses):
            views = slice(start, start + batch)
            try:
                brightness, covered = self._render_batch(vertices, faces, normals, views)
            except torch.OutOfMemoryError:
                if batch == 1:
                    raise  # even one view does not fit
                batch = max(1, batch // 2)  # free the memory and try again with half as many views
                torch.cuda.empty_cache()
                continue
            start += batch
            grey = (brightness * 255.0).round().to(torch.uint8).cpu().numpy()  # (n, H, W)
            for view_grey, mask in zip(grey, covered.cpu().numpy()):
                images.append(self._crop_view(view_grey, mask))
                touches.append(bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
                fractions.append(float(mask.mean()))
        return RenderedViews(np.stack(images), np.array(touches), np.array(fractions))

    def render_file(self, mesh_path: str | Path, front_axis: str = "+z") -> RenderedViews:
        """Load a mesh file, turn its front to +Z, normalize it to the unit sphere, and render it.

        Args:
            mesh_path: The mesh file (relative to the repository root, or absolute).
            front_axis: The direction the mesh faces in its file ("-z" for Pix3D, "+z" for ABO;
                see `render.front_axis` in `configs/render.yaml`).
        """
        vertices, faces = load_mesh(mesh_path)
        normalized, _, _ = normalize_to_unit_sphere(turn_to_front(vertices, front_axis))
        return self.render(normalized, faces)

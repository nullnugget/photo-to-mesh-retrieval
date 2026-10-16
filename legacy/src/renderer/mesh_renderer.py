"""
PyTorch3D rendering pipeline for the offline Multi-View aggregation.

Handles bounding box normalization (centering & scaling), dynamic texture 
fallbacks, and batch multi-view rendering via a fixed, canonical lighting setup.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    PointLights,
    BlendParams,
    TexturesVertex
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import IMG_SIZE, DEVICE
from src.renderer.camera_rig import build_canonical_cameras


def normalize_mesh(mesh: Meshes) -> Meshes:
    """
    Normalizes a PyTorch3D mesh by centering its bounding box at (0, 0, 0)
    and scaling its longest dimension to exactly 1.0.

    Args:
        mesh (Meshes): A PyTorch3D Meshes object (typically batch size 1).

    Returns:
        Meshes: The mathematically normalized mesh.
    """
    verts = mesh.verts_padded()
    
    # 1. Bounding Box Centering
    min_verts = verts.min(dim=1, keepdim=True).values
    max_verts = verts.max(dim=1, keepdim=True).values
    bbox_center = (min_verts + max_verts) / 2.0
    
    centered_verts = verts - bbox_center
    
    # 2. Max Dimension Scaling (fit within a 1.0 unit box)
    max_dims = (max_verts - min_verts).max(dim=2, keepdim=True).values
    scale = torch.where(max_dims == 0, torch.ones_like(max_dims), max_dims)
    
    normalized_verts = centered_verts / scale
    
    # Update the mesh geometry natively (retains existing textures/faces)
    return mesh.update_padded(normalized_verts)


class MultiViewRenderer:
    """
    Handles the batched rasterization of a normalized 3D mesh across 12 canonical views.
    """
    
    def __init__(self, img_size: int = IMG_SIZE, device: torch.device = DEVICE):
        """
        Initializes the PyTorch3D Rasterizer and Shader components.
        """
        self.device = device
        self.target_size = img_size
        self.render_size = img_size * 2  # Supersampling (448x448)
        
        self.cameras = build_canonical_cameras(distance=1.2, device=self.device)
        self.num_views = len(self.cameras)
        
        # bin_size=0 forces robust rasterization, preventing overflow crashes
        raster_settings = RasterizationSettings(
            image_size=self.render_size, 
            blur_radius=0.0, 
            faces_per_pixel=1,
            bin_size=0,
            perspective_correct=True
        )
        
        # Dynamic Lighting: Place a PointLight exactly at each of the 12 camera locations
        camera_positions = self.cameras.get_camera_center()
        self.lights = PointLights(device=self.device, location=camera_positions)
        
        # We use a white background to match our padded 2D image crops
        blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights,
                blend_params=blend_params
            )
        )

    def render(self, mesh: Meshes) -> torch.Tensor:
        """
        Takes a single mesh and renders it from all 12 canonical views.

        Args:
            mesh (Meshes): A single PyTorch3D normalized mesh on DEVICE.

        Returns:
            torch.Tensor: (12, 3, H, W) tensor of RGB renders in [0, 1] range.
        """
        # --- TEXTURE FALLBACK ---
        # If the mesh lacks texture data (e.g., missing .mtl file or dummy primitive), 
        # apply a neutral gray vertex texture to prevent shader crashes.
        if mesh.textures is None:
            verts = mesh.verts_padded()
            verts_rgb = torch.full_like(verts, 0.8, device=self.device)
            mesh.textures = TexturesVertex(verts_features=verts_rgb)

        # PyTorch3D requires the mesh batch size to match the camera batch size.
        mesh_batched = mesh.extend(self.num_views)
        
        # Render (Outputs shape: [12, H, W, 4] with RGBA channels)
        images = self.renderer(mesh_batched)
        
        # Isolate RGB channels and permute to standard PyTorch (B, C, H, W) format
        rgb_images = images[..., :3].permute(0, 3, 1, 2)

        # Anti-aliasing downsample from 448 to 224
        rgb_images = F.interpolate(
            rgb_images, 
            size=(self.target_size, self.target_size), 
            mode='bilinear', 
            align_corners=False, 
            antialias=True
        )
        
        return torch.clamp(rgb_images, 0.0, 1.0)


if __name__ == "__main__":
    print("--- Running Mesh Renderer Standalone Test ---")
    from torchvision.utils import save_image
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    
    test_out_dir = Path(__file__).resolve().parents[2] / "test_outputs"
    test_out_dir.mkdir(exist_ok=True)
    
    try:
        t_start = time.perf_counter()
        
        # 1. Define a Tetrahedron (4 vertices, 4 faces)
        verts = torch.tensor([
            [ 1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0]
        ], dtype=torch.float32, device=DEVICE)
        
        faces = torch.tensor([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ], dtype=torch.int64, device=DEVICE)
        
        # 2. Assign distinct RGB colors to the 4 vertices (Red, Green, Blue, Yellow)
        verts_rgb = torch.tensor([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [1.0, 1.0, 0.0]  
        ], dtype=torch.float32, device=DEVICE)
        
        textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))
        
        # 3. Create the mesh
        dummy_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        
        # Test Normalization
        normalized_mesh = normalize_mesh(dummy_mesh)
        verts_norm = normalized_mesh.verts_padded()
        max_coord = verts_norm.abs().max().item()
        print(f"Mesh normalized. Max coordinate boundary: {max_coord:.4f} (Expected ~0.5)")
        
        # Test Multi-View Rendering
        renderer = MultiViewRenderer()
        renders = renderer.render(normalized_mesh)
        
        t_render = time.perf_counter() - t_start
        print(f"[Timing] 12-View Batch Render: {t_render:.4f}s")
        print(f"Render tensor shape: {renders.shape}, Device: {renders.device}")
        
        # Save a grid of the 12 views
        save_image(renders, test_out_dir / "mesh_renderer_12_views.png", nrow=4)
        print(f"12 canonical views saved to {test_out_dir}")
        
    except Exception as e:
        print(f"Error during mesh rendering test: {e}")
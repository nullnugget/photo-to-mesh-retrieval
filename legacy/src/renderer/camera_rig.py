"""
Defines the 12-view canonical camera rig using icosahedral vertices.

This module guarantees reproducible, fixed viewpoints for generating 
the 3D Multi-View representations in the offline pipeline.
"""

import sys
from pathlib import Path
import math

import torch
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import DEVICE

def get_icosahedron_vertices(radius: float = 1.2, device: torch.device = DEVICE) -> torch.Tensor:
    """
    Computes the 12 vertices of a regular icosahedron.
    
    Args:
        radius (float): Distance of the cameras from the origin.
        device (torch.device): Target compute device.
        
    Returns:
        torch.Tensor: (12, 3) tensor of 3D coordinates.
    """
    phi = (1 + math.sqrt(5)) / 2
    
    # Base vertices of a regular icosahedron centered at origin
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    
    verts_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    
    # Normalize to unit length, then scale to desired radius
    verts_normalized = torch.nn.functional.normalize(verts_tensor, p=2, dim=1)
    return verts_normalized * radius

def build_canonical_cameras(distance: float = 1.2, device: torch.device = DEVICE) -> FoVPerspectiveCameras:
    """
    Constructs a PyTorch3D FoVPerspectiveCameras rig comprising 12 canonical views.
    
    Args:
        distance (float): Camera distance from the (0,0,0) origin.
        device (torch.device): Target compute device.
        
    Returns:
        FoVPerspectiveCameras: A batch of 12 cameras pointing at the origin.
    """
    camera_positions = get_icosahedron_vertices(radius=distance, device=device)
    
    # Calculate Rotation (R) and Translation (T) matrices to point at the origin (0,0,0)
    # PyTorch3D's look_at_view_transform automatically computes these based on eye positions
    R, T = look_at_view_transform(eye=camera_positions, at=((0, 0, 0),), up=((0, 1, 0),), device=device)
    
    # Initialize the batch of 12 cameras
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras

if __name__ == "__main__":
    print("--- Running Camera Rig Standalone Test ---")
    try:
        cams = build_canonical_cameras(distance=1.2)
        positions = cams.get_camera_center()
        
        print(f"Camera rig successfully initialized on {cams.device}")
        print(f"Batch size (Views): {len(cams)}")
        print(f"Camera 0 Position: {positions[0].cpu().numpy().round(3)}")
        
        # Verify distance is exactly 1.2 for all cameras
        distances = torch.norm(positions, dim=1)
        assert torch.allclose(distances, torch.tensor(1.2, device=DEVICE)), "Camera distances are uneven!"
        print("All 12 cameras are perfectly equidistant from the origin.")
        
    except Exception as e:
        print(f"Error during camera rig test: {e}")
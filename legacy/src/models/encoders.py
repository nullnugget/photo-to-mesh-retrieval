"""
Siamese Dual-Encoder architecture for 2D-to-3D Content-Based Image Retrieval.

Implements a shared ResNet50 vision backbone, a 2D linear projection head, 
and a 3D multi-view cross-attention aggregation head to map both modalities 
into a shared 512-dimensional latent space.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import WEIGHTS_DIR

class SharedBackbone(nn.Module):
    """
    Extracts 2048-dimensional feature vectors from 224x224 RGB inputs.
    Initializes a blank ResNet50 and injects locally stored pre-trained weights.
    """
    def __init__(self):
        super().__init__()
        # Initialize a blank architecture without triggering external downloads
        resnet = resnet50(weights=None)
        
        # Load local weights
        weights_path = WEIGHTS_DIR / "ResNet50" / "resnet50.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights at {weights_path}. Run scripts/download_weights.py first.")
            
        resnet.load_state_dict(torch.load(weights_path, weights_only=True))
        
        # Isolate all layers except the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (Batch, 2048, 1, 1)
        x = self.feature_extractor(x)
        # Flatten to (Batch, 2048)
        return torch.flatten(x, 1)


class ProjectionHead2D(nn.Module):
    """
    Linearly maps the 2048-dimensional 2D image feature vector into 
    the 512-dimensional shared latent space.
    """
    def __init__(self, in_features: int = 2048, out_features: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AggregationHead3D(nn.Module):
    """
    Fuses 12 view-specific feature vectors into a single 512-dimensional descriptor.
    Uses Cross-Attention where a learnable token queries the 12 views.
    """
    def __init__(self, in_features: int = 2048, out_features: int = 512, num_heads: int = 8):
        super().__init__()
        # Learnable global query token of shape (1, 1, 2048)
        self.query_token = nn.Parameter(torch.randn(1, 1, in_features))
        
        # Cross-Attention mechanism to dynamically weigh the 12 views
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Projection to shared latent space
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, 12, 2048)
        batch_size = x.size(0)
        
        # Expand query token to match batch size: (Batch, 1, 2048)
        q = self.query_token.expand(batch_size, -1, -1)
        
        # Query the 12 views. Output shape: (Batch, 1, 2048)
        attn_out, _ = self.attention(query=q, key=x, value=x)
        
        # Squeeze the sequence dimension and project
        attn_out = attn_out.squeeze(1) # (Batch, 2048)
        return self.proj(attn_out)     # (Batch, 512)


class SiameseCBIR(nn.Module):
    """
    The complete Siamese network. Routes 2D and 3D inputs through the shared 
    backbone and their respective heads, outputting L2-normalized embeddings.
    """
    def __init__(self):
        super().__init__()
        self.backbone = SharedBackbone()
        self.head_2d = ProjectionHead2D()
        self.head_3d = AggregationHead3D()

    def forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a batch of 2D queries. Input shape: (B, 3, 224, 224)"""
        features = self.backbone(x)
        embed = self.head_2d(features)
        
        # L2 normalization for cosine similarity
        return F.normalize(embed, p=2, dim=1)

    def forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a batch of 12-view 3D renders. Input shape: (B, 12, 3, 224, 224)"""
        b, v, c, h, w = x.shape
        
        # Collapse batch and view dimensions to pass through the 2D backbone
        # This step is imp to convert the input into the shape the ResNet backbone expects
        x_flat = x.view(b * v, c, h, w)
        features_flat = self.backbone(x_flat)
        
        # Unflatten back to view sequences: (B, 12, 2048)
        # During the collapse, the dimension for 12 views were merged with the batch dimension
        # Here we separate them out
        # Exach of the 12 view images is 248-dimensional
        features = features_flat.view(b, v, -1)
        
        embed = self.head_3d(features)
        
        # L2 normalization for cosine similarity
        return F.normalize(embed, p=2, dim=1)


if __name__ == "__main__":
    print("--- Running Siamese Encoder Standalone Tensor Test ---")
    
    # Simulating a strict batch size of 4 to respect the 6GB VRAM limit
    BATCH_SIZE = 4
    
    # Dummy tensors mimicking preprocessed 2D queries and 3D renders
    dummy_2d = torch.randn(BATCH_SIZE, 3, 224, 224)
    dummy_3d = torch.randn(BATCH_SIZE, 12, 3, 224, 224)
    
    print(f"Input 2D Shape: {dummy_2d.shape}")
    print(f"Input 3D Shape: {dummy_3d.shape}")
    
    # Initialize model
    model = SiameseCBIR()
    model.eval() # Disable dropout/batchnorm for deterministic testing
    
    with torch.no_grad():
        print("\nExecuting Forward Passes...")
        embed_2d = model.forward_2d(dummy_2d)
        embed_3d = model.forward_3d(dummy_3d)
        
    print(f"Output 2D Embedding Shape: {embed_2d.shape}")
    print(f"Output 3D Embedding Shape: {embed_3d.shape}")
    
    # Strict assertions for pipeline integrity
    assert embed_2d.shape == (BATCH_SIZE, 512), "2D embedding shape mismatch."
    assert embed_3d.shape == (BATCH_SIZE, 512), "3D embedding shape mismatch."
    
    # Verify L2 normalization (norm should be exactly 1.0)
    norm_2d = torch.norm(embed_2d, p=2, dim=1)
    norm_3d = torch.norm(embed_3d, p=2, dim=1)
    
    assert torch.allclose(norm_2d, torch.ones(BATCH_SIZE), atol=1e-4), "2D vectors are not L2 normalized."
    assert torch.allclose(norm_3d, torch.ones(BATCH_SIZE), atol=1e-4), "3D vectors are not L2 normalized."
    
    print("\nStandalone test passed. Shapes and L2 Normalization verified.")
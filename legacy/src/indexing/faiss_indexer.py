"""
Extract 3D gallery embeddings and build the FAISS vector index.

This script acts as the "database builder". It passes every 3D model in the dataset 
through the trained 3D encoder, extracts its 512-dimensional representation, and 
stores it in a highly optimized FAISS search index for rapid similarity lookups.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from tqdm import tqdm
import faiss

# Ensure Python can find the src/, configs/, and train.py modules at the project root
sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import PROJECT_ROOT, DEVICE
from src.models.encoders import SiameseCBIR

# ImageNet normalization constants required by ResNet50
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def load_render_views(model_dir: Path) -> torch.Tensor:
    """Loads all 12 views of a 3D model, applies RGB conversion, and normalizes them."""
    views = []
    for v in range(12):
        view_path = model_dir / f"view_{v:02d}.png"
        img = io.read_image(str(view_path), mode=io.ImageReadMode.RGB).float() / 255.0
        views.append(TF.normalize(img, mean=MEAN, std=STD))
    
    # Returns a batched tensor of shape (12, 3, 224, 224)
    return torch.stack(views)  

def build_gallery_index(is_mini: bool = False):
    renders_dir = PROJECT_ROOT / "data" / "renders"
    weights_path = PROJECT_ROOT / "weights" / "checkpoints" / "best_model.pth"

    if is_mini:
        index_out_dir = PROJECT_ROOT / "test_outputs" / "indexing"
    else:
        index_out_dir = PROJECT_ROOT / "data" / "indices"
    
    index_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading best weights from: {weights_path}")
    model = SiameseCBIR().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=DEVICE))
    model.eval()

    # Step 1: Discover all unique 3D models stored on disk
    model_paths = []
    for cat_dir in sorted(renders_dir.iterdir()):
        if not cat_dir.is_dir(): continue
        for m_dir in sorted(cat_dir.iterdir()):
            if m_dir.is_dir() and (m_dir / "view_00.png").exists():
                model_paths.append((cat_dir.name, m_dir.name, m_dir))

    # Apply --mini truncation
    if is_mini:
        model_paths = model_paths[:5]  # Process just 5 models for the dry run
        print(f"--- MINI TEST MODE: Truncated to {len(model_paths)} models ---")
    else:
        print(f"Found {len(model_paths)} unique 3D models to index.")

    embeddings = []
    metadata = []

    # Step 2: Extract 512-D embeddings for every 3D model
    with torch.no_grad():
        for cat, model_id, m_dir in tqdm(model_paths, desc="Extracting 3D Embeddings"):
            
            # Add a batch dimension to match network expectations: (1, 12, 3, 224, 224)
            views = load_render_views(m_dir).unsqueeze(0).to(DEVICE) 
            
            with torch.amp.autocast('cuda'):
                emb = model.forward_3d(views)  # Output: (1, 512)
            
            # Store the embedding and its corresponding physical identity
            embeddings.append(emb.cpu().numpy().astype(np.float32))
            metadata.append({"category": cat, "model_id": model_id})

    # Convert list of arrays into a single NumPy array of shape (N, 512)
    embeddings = np.concatenate(embeddings, axis=0)  

    # Step 3: Build FAISS Index
    # Since our network uses L2-normalization, Inner Product (IP) = Cosine Similarity.
    # IndexFlatIP performs an exact, exhaustive search, which is optimal for small datasets.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Step 4: Serialize and save to disk
    faiss_file = index_out_dir / "pix3d_3d_gallery.index"
    meta_file = index_out_dir / "gallery_metadata.json"

    faiss.write_index(index, str(faiss_file))
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved FAISS index with {index.ntotal} vectors to: {faiss_file}")
    print(f"Saved gallery metadata to: {meta_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the FAISS 3D gallery index")
    parser.add_argument('--mini', action='store_true', help="Dry-run on 5 items, saving to test_outputs/")
    args = parser.parse_args()
    
    build_gallery_index(is_mini=args.mini)
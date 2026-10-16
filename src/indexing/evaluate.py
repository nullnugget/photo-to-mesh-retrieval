"""
Evaluate 2D-to-3D cross-modal retrieval performance.

This script simulates real-world usage: it takes 2D images from the validation set, 
converts them into queries, searches the FAISS index, and compares the top-10 
retrieved 3D models against the Ground Truth (GT) to calculate benchmark metrics.
"""

import sys
import json
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
import faiss

# Ensure Python can find the root modules
sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import PROJECT_ROOT, PIX3D_JSON, DEVICE
from src.models.encoders import SiameseCBIR
from train import get_train_val_split  # Imports the deterministic splitting logic

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def evaluate():
    index_path = PROJECT_ROOT / "data" / "indices" / "pix3d_3d_gallery.index"
    meta_path = PROJECT_ROOT / "data" / "indices" / "gallery_metadata.json"
    weights_path = PROJECT_ROOT / "weights" / "checkpoints" / "best_model.pth"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    print("Loading FAISS index and gallery metadata...")
    index = faiss.read_index(str(index_path))
    with open(meta_path, 'r') as f:
        gallery_meta = json.load(f)

    # Create a reverse-lookup dictionary: map (category, model_id) -> FAISS row index
    # This allows us to quickly check if the Ground Truth matches the retrieved ID
    model_to_idx = {
        (item["category"], item["model_id"]): idx 
        for idx, item in enumerate(gallery_meta)
    }

    print("Loading Siamese model...")
    model = SiameseCBIR().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=DEVICE))
    model.eval()

    # Leverage the strict random seed to recreate the exact 15% validation split
    _, val_meta = get_train_val_split(PIX3D_JSON)

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    cat_correct = 0
    valid_queries = 0

    with torch.no_grad():
        for item in tqdm(val_meta, desc="Evaluating Queries"):
            img_path = (processed_dir / item['img']).with_suffix('.png')
            
            # Skip missing data gracefully
            if not img_path.exists(): continue

            # Determine the Ground Truth (GT) identity of this 2D image
            gt_cat = item['category']
            gt_model = Path(item['model']).parent.name

            # Skip if the GT 3D model wasn't successfully indexed in the gallery
            if (gt_cat, gt_model) not in model_to_idx:
                continue
            gt_idx = model_to_idx[(gt_cat, gt_model)]

            # Load and normalize the 2D query
            img = io.read_image(str(img_path), mode=io.ImageReadMode.RGB).float() / 255.0
            img = TF.normalize(img, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)

            # Generate the 512-D query embedding
            with torch.amp.autocast('cuda'):
                q_emb = model.forward_2d(img).cpu().numpy().astype(np.float32)

            # Search the FAISS index for the k=10 most similar 3D embeddings
            _, indices = index.search(q_emb, k=10)
            retrieved_indices = indices[0].tolist()

            # --- Calculate Metrics ---
            # Instance-Level Recall: Did we retrieve the exact same piece of furniture?
            if gt_idx == retrieved_indices[0]: top1_correct += 1
            if gt_idx in retrieved_indices[:5]: top5_correct += 1
            if gt_idx in retrieved_indices[:10]: top10_correct += 1

            # Semantic Accuracy: Did the #1 result at least belong to the same category?
            pred_cat = gallery_meta[retrieved_indices[0]]["category"]
            if pred_cat == gt_cat: cat_correct += 1

            valid_queries += 1

    print("\n" + "=" * 45)
    print(f"Validation Evaluation Summary ({valid_queries} queries)")
    print("=" * 45)
    print(f"Recall@1:             {top1_correct / valid_queries * 100:.2f}%")
    print(f"Recall@5:             {top5_correct / valid_queries * 100:.2f}%")
    print(f"Recall@10:            {top10_correct / valid_queries * 100:.2f}%")
    print(f"Category Accuracy@1:  {cat_correct / valid_queries * 100:.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    evaluate()

# =============================================
# Validation Evaluation Summary (1510 queries)
# =============================================
# Recall@1:             25.03%
# Recall@5:             56.49%
# Recall@10:            70.33%
# Category Accuracy@1:  77.35%
# =============================================
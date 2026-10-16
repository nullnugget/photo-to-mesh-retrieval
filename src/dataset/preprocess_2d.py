"""
Offline execution script to preprocess all 2D query images in the Pix3D dataset.

Uses the ground-truth binary masks provided by the Pix3D dataset to perfectly 
isolate the objects, crops them using the provided bounding boxes, pads them 
into a square with a white background, and resizes them to 224x224.
"""

import sys
import json
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import PROJECT_ROOT, PIX3D_DIR, PIX3D_JSON

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("Preprocess2D")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    return logger

def get_target_indices(metadata: list, mode: str) -> list:
    if mode == 'full':
        return list(range(len(metadata)))
    if mode == 'test':
        return [0]
    if mode == 'mini':
        cat_to_indices = defaultdict(list)
        for idx, item in enumerate(metadata):
            cat_to_indices[item['category']].append(idx)
        indices = []
        for cat, idxs in cat_to_indices.items():
            indices.extend(idxs[:5])
        return indices

def process_ground_truth_crop(image: torch.Tensor, mask: torch.Tensor, bbox: list, target_size: int = 224) -> torch.Tensor:
    """Applies a ground-truth mask, crops, square-pads, and resizes."""
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
        
    # Ensure mask is boolean and matches image spatial dimensions
    # Mask is usually 1 channel. > 127 treats white as True (foreground)
    mask_bool = mask[0:1] > 127 
    
    # 1. Apply Mask (Set background to pure white)
    white_bg = torch.ones_like(image)
    masked_image = torch.where(mask_bool, image, white_bg)
    
    # 2. Crop to Bounding Box
    x1, y1, x2, y2 = [int(v) for v in bbox]
    _, h, w = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cropped = masked_image[:, y1:y2, x1:x2]
    
    # 3. Pad to Square
    crop_h, crop_w = cropped.shape[1:]
    max_dim = max(crop_h, crop_w)
    
    pad_h = max_dim - crop_h
    pad_w = max_dim - crop_w
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    squared = torch.nn.functional.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
    
    # 4. Resize and clamp
    resized = TF.resize(squared, [target_size, target_size], antialias=True)
    return resized.clamp(0.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Preprocess 2D Queries using Ground Truth Masks")
    parser.add_argument('--full', action='store_true', help="Process all images in the dataset")
    parser.add_argument('--mini', action='store_true', help="Process 5 images per class")
    args = parser.parse_args()

    mode = 'full' if args.full else 'mini' if args.mini else 'test'
    out_dir = PROJECT_ROOT / "data" / "processed" if args.full else PROJECT_ROOT / "test_outputs" / f"processed_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'test':
        print("--- Running Preprocess 2D Standalone Test (Ground Truth) ---")

    logger = setup_logger(out_dir / "preprocess_2d.log")
    
    with open(PIX3D_JSON, 'r') as f:
        metadata = json.load(f)

    indices = get_target_indices(metadata, mode)
    pbar = tqdm(indices, desc="Processing 2D Queries", unit="img")
    
    success_count, fail_count = 0, 0

    for idx in pbar:
        item = metadata[idx]
        rel_img_path = Path(item['img'])
        
        raw_img_path = PIX3D_DIR / rel_img_path
        mask_path = PIX3D_DIR / item['mask']
        
        target_path = (out_dir / rel_img_path).with_suffix('.png')
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and mode != 'test':
            success_count += 1
            continue

        try:
            image = io.read_image(str(raw_img_path))
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                image = image[:3]
                
            mask = io.read_image(str(mask_path))
            
            final_224 = process_ground_truth_crop(image, mask, item['bbox'], target_size=224)
            save_image(final_224, target_path)
            
            success_count += 1
            
            if mode == 'test':
                print(f"Standalone test passed. Image saved to {target_path}")
                break
                
        except Exception as e:
            logger.error(f"FAILED - {rel_img_path} | Error: {str(e)}")
            fail_count += 1

    if mode != 'test':
        print(f"\Preprocessing complete. Success: {success_count} | Failed: {fail_count}")

if __name__ == "__main__":
    main()
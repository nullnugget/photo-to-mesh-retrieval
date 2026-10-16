"""
Point-prompted image segmentation using FastSAM.

This module provides a wrapper around the FastSAM model to extract binary masks 
and a utility to crop, pad, and resize 2D images to match the canonical 3D renders.

The extration is handled entirely on GPU using pytorch. 
"""

"""
Bounding-box prompted image segmentation and extraction using FastSAM.

Provides a wrapper around FastSAM to extract binary masks and a utility 
to crop, pad, and resize 2D images to match the canonical 3D renders.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, draw_segmentation_masks

try:
    from ultralytics import FastSAM
except ImportError:
    raise ImportError("The 'ultralytics' package is required. Install via: pip install ultralytics")

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import WEIGHTS_DIR, DEVICE, PROJECT_ROOT

class FastSAMSegmenter:
    """
    Wrapper for FastSAM to handle object segmentation entirely on-device.
    """
    
    def __init__(self, weights_filename: str = "FastSAM-s.pt"):
        weights_path = WEIGHTS_DIR / "FastSAM" / weights_filename
        if not weights_path.exists():
            raise FileNotFoundError(f"FastSAM weights not found at {weights_path}")
            
        self.device = DEVICE
        self.model = FastSAM(str(weights_path))
    
    def _prepare_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Pads image to stride 32 and normalizes for Ultralytics."""
        c, h, w = image.shape
        stride = 32
        pad_w = (stride - w % stride) % stride
        pad_h = (stride - h % stride) % stride
        
        if pad_w > 0 or pad_h > 0:
            image_padded = F.pad(image, (0, pad_w, 0, pad_h), value=0)
        else:
            image_padded = image
            
        img_batch = image_padded.unsqueeze(0).to(self.device)
        
        if img_batch.dtype == torch.uint8:
            img_batch = img_batch.float() / 255.0
            
        return img_batch, h, w

    def segment_by_point(
        self, image: torch.Tensor, point: Tuple[int, int], conf: float = 0.4, iou: float = 0.9
    ) -> Optional[torch.Tensor]:
        """
        Generates a binary mask natively on the GPU, with dynamic stride padding.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W) on DEVICE.
            point (Tuple[int, int]): (x, y) coordinates of the point prompt.
            img_size (int): Internal resolution for the FastSAM forward pass.
            conf (float): Confidence threshold for segmentation.
            iou (float): Intersection over Union threshold for NMS.

        Returns:
            Optional[torch.Tensor]: Boolean mask of shape (H, W) on DEVICE, 
                                    or None if no object is detected.
        """
        
        img_batch, h, w = self._prepare_image(image)
        
        results = self.model.predict(
            source=img_batch, device=self.device, retina_masks=True, 
            conf=conf, iou=iou, points=[[point[0], point[1]]], labels=[1], verbose=False
        )
        
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            return None
            
        mask_tensor = result.masks.data[0].bool()
        return mask_tensor[:h, :w]

    def segment_by_bbox(
        self, image: torch.Tensor, bbox: List[float], conf: float = 0.4, iou: float = 0.9
    ) -> Optional[torch.Tensor]:
        """
        Generates a binary mask based on a bounding box prompt [x1, y1, x2, y2].
        """
        img_batch, h, w = self._prepare_image(image)
        
        results = self.model.predict(
            source=img_batch, device=self.device, retina_masks=True, 
            conf=conf, iou=iou, bboxes=[bbox], verbose=False
        )
        
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            return None
            
        mask_tensor = result.masks.data[0].bool()
        return mask_tensor[:h, :w]

    def extract_and_pad(
        self, image: torch.Tensor, mask: torch.Tensor, bbox: List[float], target_size: int = 224
    ) -> torch.Tensor:
        """
        Applies mask, crops to bbox, pads to a square (preserving aspect ratio), 
        and resizes to target_size. Uses a white background.
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
            
        # 1. Apply Mask (Set background to pure white)
        white_bg = torch.ones_like(image)
        masked_image = torch.where(mask.unsqueeze(0), image, white_bg)
        
        # 2. Crop to Bounding Box
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Clamp coordinates to image boundaries
        c, h, w = image.shape
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
        
        # F.pad expects (left, right, top, bottom). value=1.0 is white.
        squared = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
        
        # 4. Resize to target size (224x224)
        resized = TF.resize(squared, [target_size, target_size], antialias=True)
        return resized


if __name__ == "__main__":
    print("--- Running FastSAMSegmenter Standalone Test ---")
    
    test_out_dir = PROJECT_ROOT / "test_outputs"
    test_out_dir.mkdir(exist_ok=True)
    
    segmenter = FastSAMSegmenter()
    
    # Generate a noisy background image
    dummy_image = torch.randint(100, 200, (3, 512, 512), dtype=torch.uint8, device=DEVICE)
    
    # Draw a pure red rectangle (our "object")
    dummy_image[0, 150:350, 100:400] = 255
    dummy_image[1, 150:350, 100:400] = 0
    dummy_image[2, 150:350, 100:400] = 0
    
    # 1. Test BBox Segmentation
    bbox = [100.0, 150.0, 400.0, 350.0]
    print("\n--- Testing BBox Prompt ---")
    try:
        t_start = time.perf_counter()
        bbox_mask = segmenter.segment_by_bbox(dummy_image, bbox)
        print(f"BBox Segmentation: {time.perf_counter() - t_start:.4f}s")
        
        if bbox_mask is not None:
            final_224 = segmenter.extract_and_pad(dummy_image, bbox_mask, bbox)
            save_image(final_224, test_out_dir / "bbox_final_224.png")
            print(f"BBox image saved to: {test_out_dir / 'bbox_final_224.png'}")
        else:
            print("BBox model executed but returned no mask.")
    except Exception as e:
        print(f"Error during BBox test: {e}")

    # 2. Test Point Segmentation
    target_point = (250, 250)
    print("\n--- Testing Point Prompt ---")
    try:
        t_start = time.perf_counter()
        point_mask = segmenter.segment_by_point(dummy_image, target_point)
        print(f"Point Segmentation: {time.perf_counter() - t_start:.4f}s")
        
        if point_mask is not None:
            overlay = draw_segmentation_masks(dummy_image.cpu(), point_mask.cpu(), alpha=0.5, colors=["blue"])
            save_image(overlay.float() / 255.0, test_out_dir / "point_masked.png")
            print(f"Point mask overlay saved to: {test_out_dir / 'point_masked.png'}")
        else:
            print("Point model executed but returned no mask.")
    except Exception as e:
        print(f"Error during Point test: {e}")
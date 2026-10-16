"""
Image manipulation utilities for 2D preprocessing.

Handles bounding box extraction, object cropping, square aspect-ratio padding,
and resizing for neural network inputs.
"""

"""
Image manipulation utilities for 2D preprocessing.

Handles bounding box extraction, object cropping, square aspect-ratio padding,
and resizing natively on the GPU (using PyTorch).
"""

import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, draw_bounding_boxes

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import IMG_SIZE, PROJECT_ROOT, DEVICE

def extract_bounding_box(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Computes the tightest bounding box around the True values in a binary mask using PyTorch.

    Args:
        mask (torch.Tensor): Binary mask of shape (H, W) on GPU.

    Returns:
        Tuple[int, int, int, int]: (x, y, width, height) of the bounding box.
                                   Returns (0, 0, 0, 0) if mask is empty.
    """
    nonzero_indices = mask.nonzero()
    
    if nonzero_indices.numel() == 0:
        return 0, 0, 0, 0
        
    y_min, x_min = nonzero_indices.min(dim=0).values.tolist()
    y_max, x_max = nonzero_indices.max(dim=0).values.tolist()
    
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    return x_min, y_min, w, h

def crop_and_pad(
    image: torch.Tensor, 
    mask: torch.Tensor, 
    target_size: int = IMG_SIZE, 
    pad_color: Tuple[int, int, int] = (255, 255, 255),
    margin: int = 10
) -> torch.Tensor:
    """
    Crops the object using the mask bounding box, pads it to a square, 
    and resizes it to the target resolution.

    Args:
        image (torch.Tensor): Image tensor (C, H, W) on GPU.
        mask (torch.Tensor): Binary mask array (H, W) on GPU.
        target_size (int): Resolution of the output square image.
        pad_color (Tuple[int, int, int]): RGB color for padding and background.
        margin (int): Pixel margin to leave around the bounding box before padding.

    Returns:
        torch.Tensor: Cropped, padded, and resized RGB image (C, target_size, target_size).
    """
    _, img_h, img_w = image.shape
    x, y, w, h = extract_bounding_box(mask)
    
    pad_color_tensor = torch.tensor(pad_color, dtype=image.dtype, device=image.device).view(3, 1, 1)

    if w == 0 or h == 0:
        return pad_color_tensor.expand(3, target_size, target_size)

    # 1. Mask out background (apply pad color to pixels outside mask)
    masked_img = torch.where(mask.unsqueeze(0), image, pad_color_tensor)

    # 2. Apply margin and clamp to borders
    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(img_w, x + w + margin)
    y_max = min(img_h, y + h + margin)

    # 3. Crop
    crop_img = masked_img[:, y_min:y_max, x_min:x_max]

    # 4. Pad to square
    _, crop_h, crop_w = crop_img.shape
    max_side = max(crop_h, crop_w)
    
    pad_top = (max_side - crop_h) // 2
    pad_bottom = max_side - crop_h - pad_top
    pad_left = (max_side - crop_w) // 2
    pad_right = max_side - crop_w - pad_left

    # F.pad format is (pad_left, pad_right, pad_top, pad_bottom)
    # The value must be a single scalar in F.pad. Since we have a 3-channel pad_color, 
    # and standard Pix3D padding is usually uniform white or black, we pad with the first channel 
    # and rely on the background masking done in step 1 to handle internal space. 
    # Assuming pad_color is grayscale (e.g. 255, 255, 255 or 0, 0, 0):
    square_img = F.pad(crop_img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_color[0])

    # 5. Resize using torchvision
    resized_img = TF.resize(square_img, [target_size, target_size], antialias=True)
    
    return resized_img

if __name__ == "__main__":
    print("--- Running Image Utils Standalone Test ---")
    
    test_out_dir = PROJECT_ROOT / "test_outputs"
    test_out_dir.mkdir(exist_ok=True)
    
    t_start = time.perf_counter()
    
    # Dummy image (C, H, W) and mask on GPU
    dummy_img = torch.full((3, 512, 512), 255, dtype=torch.uint8, device=DEVICE)
    dummy_img[0, 100:300, 200:300] = 255 # Red bounding box area
    dummy_img[1, 100:300, 200:300] = 0
    dummy_img[2, 100:300, 200:300] = 0
    
    dummy_mask = torch.zeros((512, 512), dtype=torch.bool, device=DEVICE)
    dummy_mask[100:300, 200:300] = True
    
    t_setup = time.perf_counter() - t_start
    print(f"[Timing] Setup & GPU Transfer: {t_setup:.6f}s")
    
    try:
        # Test 1: Bounding Box
        t_start = time.perf_counter()
        bbox = extract_bounding_box(dummy_mask)
        t_bbox = time.perf_counter() - t_start
        print(f"[Timing] BBox Extraction:    {t_bbox:.6f}s")
        print(f"Bounding box: {bbox}")
        
        # Test 2: Crop & Pad
        t_start = time.perf_counter()
        processed = crop_and_pad(dummy_img, dummy_mask, target_size=224, pad_color=(255, 255, 255))
        t_crop = time.perf_counter() - t_start
        print(f"[Timing] Crop, Pad, Resize:  {t_crop:.6f}s")
        print(f"Processed shape: {processed.shape}, Device: {processed.device}")
        
        # Visualization Phase (Offload to CPU strictly for saving)
        t_start = time.perf_counter()
        
        # Draw bounding box on original image
        x, y, w, h = bbox
        boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float)
        img_with_bbox = draw_bounding_boxes(dummy_img.cpu(), boxes, colors=["blue"], width=3)
        
        save_image(dummy_img.float() / 255.0, test_out_dir / "utils_01_original.png")
        save_image(img_with_bbox.float() / 255.0, test_out_dir / "utils_02_with_bbox.png")
        save_image(processed.float() / 255.0, test_out_dir / "utils_03_cropped_padded.png")
        
        t_save = time.perf_counter() - t_start
        print(f"[Timing] CPU Offload & Save: {t_save:.6f}s")
        print(f"Test outputs saved to {test_out_dir}")
        
    except Exception as e:
        print(f"Error during image utils test: {e}")

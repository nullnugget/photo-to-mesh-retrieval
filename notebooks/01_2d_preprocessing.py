import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    import torch
    import torchvision.transforms.functional as TF
    from torchvision.utils import draw_keypoints, draw_segmentation_masks
    from PIL import Image
    import json

    # Resolve project root and append to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))

    from configs.config import PIX3D_DIR, PIX3D_JSON, DEVICE
    from src.segmentation.segmenter import FastSAMSegmenter
    from src.utils.image_utils import crop_and_pad, extract_bounding_box

    return (
        DEVICE,
        FastSAMSegmenter,
        Image,
        PIX3D_DIR,
        PIX3D_JSON,
        TF,
        crop_and_pad,
        draw_keypoints,
        draw_segmentation_masks,
        json,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 1: 2D Preprocessing & Segmentation Pipeline

    This dashboard validates our custom utilities on the Pix3D dataset. We will test bounding box extraction, square-padding, and FastSAM point-prompted segmentation.
    """)
    return


@app.cell
def _(PIX3D_JSON, json, mo):
    # Load Pix3D Metadata
    with open(PIX3D_JSON, 'r') as f:
        metadata = json.load(f)
    
    # Extract unique categories
    categories = sorted(list(set(item['category'] for item in metadata)))

    # UI: Select Category
    category_dropdown = mo.ui.dropdown(
        options=categories, 
        value="chair", 
        label="1. Select Object Category:"
    )

    # Create the dropdown button
    category_dropdown
    return category_dropdown, metadata


@app.cell
def _(category_dropdown, metadata, mo):
    # Filter dataset by selected category
    filtered_data = [item for item in metadata if item['category'] == category_dropdown.value]

    # UI: Select specific image by index
    idx_slider = mo.ui.slider(
        start=0, 
        stop=len(filtered_data) - 1, 
        step=1, 
        value=0, 
        label="2. Select Image Index:"
    )

    idx_slider
    return filtered_data, idx_slider


@app.cell
def _(FastSAMSegmenter):
    # Initialize the Segmenter once (caches the model on GPU)
    segmenter = FastSAMSegmenter()
    return (segmenter,)


@app.cell
def _(Image, PIX3D_DIR, TF, filtered_data, idx_slider, torch):
    # Load the selected image and ground truth mask
    selected_item = filtered_data[idx_slider.value]

    img_path = PIX3D_DIR / selected_item['img']
    mask_path = PIX3D_DIR / selected_item['mask']

    # Load as PIL, convert to tensor (C, H, W) in [0, 255] range
    raw_pil = Image.open(img_path).convert("RGB")
    raw_tensor = (TF.to_tensor(raw_pil) * 255).to(torch.uint8)

    gt_mask_pil = Image.open(mask_path).convert("L")
    # Ground truth mask thresholding
    gt_mask_tensor = (TF.to_tensor(gt_mask_pil) > 0.5).squeeze(0)

    _, img_h, img_w = raw_tensor.shape
    return gt_mask_tensor, img_h, img_w, raw_tensor


@app.cell
def _(img_h, img_w, mo):
    # UI: Coordinate sliders to act as our "click"
    x_slider = mo.ui.slider(start=0, stop=img_w-1, value=img_w//2, label="X Coordinate")
    y_slider = mo.ui.slider(start=0, stop=img_h-1, value=img_h//2, label="Y Coordinate")

    mo.hstack([x_slider, y_slider])
    return x_slider, y_slider


@app.cell
def _(
    DEVICE,
    TF,
    crop_and_pad,
    draw_keypoints,
    draw_segmentation_masks,
    gt_mask_tensor,
    mo,
    raw_tensor,
    segmenter,
    torch,
    x_slider,
    y_slider,
):
    # 1. Prepare Target Point
    target_pt = (x_slider.value, y_slider.value)
    pt_tensor = torch.tensor([[target_pt]], dtype=torch.float)

    # 2. Draw Crosshair on Original Image
    img_with_pt = draw_keypoints(raw_tensor, pt_tensor, colors="red", radius=5)

    # 3. Run FastSAM Prediction (GPU)
    fastsam_mask = segmenter.segment_by_point(
        raw_tensor.to(DEVICE), 
        point=target_pt
    )

    # 4. Generate Cropped & Padded Outputs
    # A) Using Ground Truth Mask
    gt_processed = crop_and_pad(
        raw_tensor.to(DEVICE), 
        gt_mask_tensor.to(DEVICE), 
        target_size=224
    )

    # B) Using FastSAM Mask
    if fastsam_mask is not None:
        fs_processed = crop_and_pad(
            raw_tensor.to(DEVICE), 
            fastsam_mask, 
            target_size=224
        )
        # Overlay mask for visualization
        fs_overlay = draw_segmentation_masks(raw_tensor, fastsam_mask.cpu(), alpha=0.5, colors=["blue"])
    else:
        fs_processed = torch.zeros((3, 224, 224), dtype=torch.uint8)
        fs_overlay = raw_tensor.clone()

    # 5. Display the Visuals side-by-side using Marimo's UI
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("**Target Point (Red)**"), mo.image(TF.to_pil_image(img_with_pt))]),
            mo.vstack([mo.md("**FastSAM Mask Overlay**"), mo.image(TF.to_pil_image(fs_overlay))])
        ]),
        mo.md("---"),
        mo.hstack([
            mo.vstack([mo.md("**Cropped via Ground Truth**"), mo.image(TF.to_pil_image(gt_processed))]),
            mo.vstack([mo.md("**Cropped via FastSAM**"), mo.image(TF.to_pil_image(fs_processed))])
        ])
    ])
    return


if __name__ == "__main__":
    app.run()

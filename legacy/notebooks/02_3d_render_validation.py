import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import json
    import random
    from pathlib import Path

    import marimo as mo
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.utils import make_grid

    # PyTorch3D mesh loading
    from pytorch3d.io import load_objs_as_meshes

    # Resolve project root and append to sys.path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))

    from configs.config import PIX3D_DIR, PIX3D_JSON, DEVICE
    from src.renderer.mesh_renderer import MultiViewRenderer, normalize_mesh

    return (
        DEVICE,
        MultiViewRenderer,
        PIX3D_DIR,
        PIX3D_JSON,
        TF,
        json,
        load_objs_as_meshes,
        make_grid,
        mo,
        normalize_mesh,
        random,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Phase 2: 3D Multi-View Rendering Validation
    This dashboard validates our PyTorch3D canonical camera rig, mesh normalization math, and lighting setup.
    We ensure aggressive memory management (`torch.cuda.empty_cache()`) to respect the 6GB VRAM limit.
    """)
    return


@app.cell
def _(MultiViewRenderer, PIX3D_JSON, json):
    # Load Metadata and initialize the Renderer once
    with open(PIX3D_JSON, 'r') as f:
        metadata = json.load(f)

    categories = sorted(list(set(item['category'] for item in metadata)))

    renderer = MultiViewRenderer()
    return categories, metadata, renderer


@app.cell
def _(categories, mo):
    mo.md("## 1. Single Object Explorer")

    explorer_category = mo.ui.dropdown(
        options=categories, 
        value="chair", 
        label="Select Category:"
    )
    explorer_category
    return (explorer_category,)


@app.cell
def _(explorer_category, metadata, mo):
    # Filter dataset for the explorer
    explorer_data = [item for item in metadata if item['category'] == explorer_category.value]

    explorer_idx = mo.ui.slider(
        start=0, 
        stop=len(explorer_data) - 1, 
        step=1, 
        value=0, 
        label="Select Object Index:"
    )
    explorer_idx
    return explorer_data, explorer_idx


@app.cell
def _(
    DEVICE,
    PIX3D_DIR,
    TF,
    explorer_data,
    explorer_idx,
    load_objs_as_meshes,
    make_grid,
    mo,
    normalize_mesh,
    renderer,
    torch,
):
    # Execute Single Object Render
    selected_item = explorer_data[explorer_idx.value]
    obj_path = PIX3D_DIR / selected_item['model']

    try:
        # 1. Load mesh
        mesh = load_objs_as_meshes(
            [obj_path], 
            device=DEVICE, 
            load_textures=True, 
            create_texture_atlas=True
        )

        # 2. Normalize and Render
        norm_mesh = normalize_mesh(mesh)
        renders = renderer.render(norm_mesh) # Shape: (12, 3, 224, 224)

        # 3. Create a clean 3x4 image grid
        grid = make_grid(renders, nrow=4, padding=2, pad_value=0.8)

        # 4. Memory Cleanup - Crucial for slider interactivity
        del mesh, norm_mesh, renders
        torch.cuda.empty_cache()

        # Display
        single_display = mo.vstack([
            mo.md(f"**Model Path:** `{selected_item['model']}`"),
            mo.image(TF.to_pil_image(grid))
        ])
    except Exception as e:
        single_display = mo.md(f"**Error rendering mesh:** {e}")

    single_display
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 2. Multi-Class Validation Matrix
    Click the button below to randomly select 1 object from **every** category and render all 12 views.
    *Note: This takes a few seconds. Meshes are processed sequentially and flushed from VRAM to prevent memory crashes.*
    """)
    return


@app.cell
def _(mo):
    generate_matrix_btn = mo.ui.button(label="Generate Multi-Class Matrix", kind="success")
    generate_matrix_btn
    return (generate_matrix_btn,)


@app.cell
def _(
    DEVICE,
    PIX3D_DIR,
    TF,
    categories,
    generate_matrix_btn,
    load_objs_as_meshes,
    make_grid,
    metadata,
    mo,
    normalize_mesh,
    random,
    renderer,
    torch,
):
    matrix_display = mo.md("Click the button to generate.")

    if generate_matrix_btn.value:
        all_class_renders = []

        with mo.status.spinner(title="Generating Multi-Class Matrix..."):
            for matrix_cat in categories:
                # Get random item from this category
                matrix_cat_items = [item for item in metadata if item['category'] == matrix_cat]
                matrix_rand_item = random.choice(matrix_cat_items)
                matrix_obj_path = PIX3D_DIR / matrix_rand_item['model']
        
                try:
                    # Sequential load -> normalize -> render
                    matrix_mesh = load_objs_as_meshes(
                            [matrix_obj_path], 
                            device=DEVICE, 
                            load_textures=True, 
                            create_texture_atlas=True
                        )
                    matrix_norm_mesh = normalize_mesh(matrix_mesh)
            
                    # Render and immediately move the 12 images to CPU RAM
                    matrix_renders = renderer.render(matrix_norm_mesh).cpu() 
                    all_class_renders.append(matrix_renders)
            
                    # Aggressive cleanup before the next category loads
                    del matrix_mesh, matrix_norm_mesh
                    torch.cuda.empty_cache()
            
                except Exception as e:
                    print(f"Skipped {matrix_cat} due to error: {e}")
                    continue

        if all_class_renders:
            # Concatenate all CPU tensors: [categories * 12 views]
            matrix_tensor = torch.cat(all_class_renders, dim=0)
    
            # Make a grid where each row represents one category (nrow=12 columns)
            huge_grid = make_grid(matrix_tensor, nrow=12, padding=4, pad_value=0.5)
            matrix_display = mo.image(TF.to_pil_image(huge_grid))
        else:
            matrix_display = mo.md("Failed to generate any renders.")

    matrix_display
    return


if __name__ == "__main__":
    app.run()

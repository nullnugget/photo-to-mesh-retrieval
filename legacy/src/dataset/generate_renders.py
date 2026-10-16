"""
Offline execution script to generate 12-view 3D renders for all unique Pix3D models.

Implements sequential batch processing, aggressive GPU memory management, 
CLI run modes (--full, --mini), and file-based logging.
"""

import sys
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
from torchvision.utils import save_image
from pytorch3d.io import load_objs_as_meshes
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import PROJECT_ROOT, DEVICE
from src.dataset.pix3d_loader import Pix3DUniqueModelsDataset
from src.renderer.mesh_renderer import MultiViewRenderer, normalize_mesh

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("GenerateRenders")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(fh)
    
    return logger

def get_target_indices(dataset: Pix3DUniqueModelsDataset, mode: str) -> list:
    """Returns a list of dataset indices to process based on the run mode."""
    if mode == 'full':
        return list(range(len(dataset)))
        
    if mode == 'test':
        return [0]
        
    if mode == 'mini':
        cat_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.unique_models):
            cat_to_indices[item['category']].append(idx)
            
        indices = []
        for cat, idxs in cat_to_indices.items():
            # Updated to 5 objects per class
            indices.extend(idxs[:5])
        return indices

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Renders for Pix3D")
    parser.add_argument('--full', action='store_true', help="Process all unique CAD models")
    parser.add_argument('--mini', action='store_true', help="Process 5 models per class for pipeline verification")
    args = parser.parse_args()
    
    # Safely route outputs based on the run mode
    if args.full:
        mode = 'full'
        render_out_dir = PROJECT_ROOT / "data" / "renders"
    elif args.mini:
        mode = 'mini'
        render_out_dir = PROJECT_ROOT / "test_outputs" / "renders_mini"
    else:
        mode = 'test'
        render_out_dir = PROJECT_ROOT / "test_outputs" / "renders_test"
        print("⚠️ No flags provided. Defaulting to 'test' mode (1 object). Use --mini or --full for broader runs.")

    render_out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(render_out_dir / "generation.log")
    logger.info(f"Starting generation script in '{mode}' mode.")
    
    dataset = Pix3DUniqueModelsDataset()
    indices_to_process = get_target_indices(dataset, mode)
    
    logger.info(f"Targeting {len(indices_to_process)} meshes for processing.")
    
    renderer = MultiViewRenderer()
    success_count = 0
    fail_count = 0
    
    pbar = tqdm(indices_to_process, desc="Rendering Multi-View", unit="mesh")
    
    for idx in pbar:
        model_path, category, model_id = dataset[idx]
        
        target_dir = render_out_dir / category / model_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if (target_dir / "view_11.png").exists():
            logger.info(f"Skipping {category}/{model_id} - Already exists.")
            success_count += 1
            continue
            
        try:
            t_mesh_start = time.perf_counter()

            mesh = load_objs_as_meshes(
                [model_path], 
                device=DEVICE, 
                load_textures=True, 
                create_texture_atlas=True
            )
            
            norm_mesh = normalize_mesh(mesh)
            renders = renderer.render(norm_mesh) 
            renders_cpu = renders.cpu()
            
            for view_idx in range(12):
                img_tensor = renders_cpu[view_idx]
                out_file = target_dir / f"view_{view_idx:02d}.png"
                save_image(img_tensor, out_file)
                
            del mesh, norm_mesh, renders, renders_cpu
            torch.cuda.empty_cache()

            t_mesh_end = time.perf_counter()

            logger.info(f"SUCCESS - {category}/{model_id} | Time: {t_mesh_end - t_mesh_start:.2f}s")
            success_count += 1
            
        except Exception as e:
            logger.error(f"FAILED - {category}/{model_id} | Error: {str(e)}")
            fail_count += 1
            torch.cuda.empty_cache()

    logger.info(f"--- GENERATION COMPLETE ({mode.upper()}) ---")
    logger.info(f"Successful: {success_count} | Failed: {fail_count}")
    print(f"\nGeneration complete. Success: {success_count} | Failed: {fail_count}")
    print(f"Check {render_out_dir / 'generation.log'} for detailed execution logs.")

if __name__ == "__main__":
    main()
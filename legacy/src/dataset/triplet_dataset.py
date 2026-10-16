"""
PyTorch Dataset and Stratified Sampler for 2D-to-3D Cross-Modal Retrieval.

Loads the 2D queries and their corresponding 12-view 3D renders. Applies 
ImageNet normalization and ensures batches contain diverse categories to 
maximize the effectiveness of the InfoNCE contrastive loss.
"""

import sys
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Iterator, List

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import torchvision.io as io
import torchvision.transforms.functional as TF

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import PROJECT_ROOT, PIX3D_JSON, SEED

class Pix3DContrastiveDataset(Dataset):
    """
    Loads 2D queries and 12-view 3D renders, applying ImageNet normalization.
    """
    def __init__(self, metadata_or_path, processed_2d_dir: Path, renders_3d_dir: Path):
        self.processed_2d_dir = processed_2d_dir
        self.renders_3d_dir = renders_3d_dir
        
        # Accept either a path to the JSON or a direct list of pre-filtered dictionaries
        if isinstance(metadata_or_path, (str, Path)):
            with open(metadata_or_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = metadata_or_path
            
        # ImageNet standardization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Filter metadata to only include items where both 2D and 3D data exist on disk
        self.valid_items = self._validate_dataset()

    def _validate_dataset(self) -> List[dict]:
        valid = []
        for item in self.metadata:
            img_path = (self.processed_2d_dir / item['img']).with_suffix('.png')
            model_id = Path(item['model']).parent.name
            cat = item['category']
            
            # Check if all 12 views exist (Filename Format: 'view_XX.png')
            views_exist = all(
                (self.renders_3d_dir / cat / model_id / f"view_{v:02d}.png").exists() 
                for v in range(12)
            )
            
            if img_path.exists() and views_exist:
                valid.append(item)
                
        return valid

    def _load_and_normalize(self, path: Path) -> torch.Tensor:
        # IMP - Enforce exactly 3 channels (RGB) regardless of the original file format
        img = io.read_image(str(path), mode=io.ImageReadMode.RGB)
        img = img.float() / 255.0
        return TF.normalize(img, mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.valid_items)

    def __getitem__(self, idx):
        item = self.valid_items[idx]
        cat = item['category']
        
        try:
            # 1. Load 2D Query
            img_path = (self.processed_2d_dir / item['img']).with_suffix('.png')
            img_2d = self._load_and_normalize(img_path)
            
            # 2. Load 3D Positive Anchors (12 Views)
            model_id = Path(item['model']).parent.name
            views_3d = []
            for v in range(12):
                view_path = self.renders_3d_dir / cat / model_id / f"view_{v:02d}.png"
                views_3d.append(self._load_and_normalize(view_path))
                
            # Stack to (12, 3, 224, 224)
            tensor_3d = torch.stack(views_3d)
            
            return img_2d, tensor_3d, cat
            
        except Exception as e:
            # Intercept the crash and expose the exact file causing the dimension mismatch
            raise RuntimeError(f"Dataloader crashed while processing 2D query: {img_path} or its associated 3D renders.") from e


class StratifiedBatchSampler(Sampler):
    """
    Ensures each batch contains a diverse mix of categories to provide 
    strong negative samples for the contrastive loss function.
    """
    def __init__(self, dataset: Pix3DContrastiveDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Enforce reproducibility for the sampler
        random.seed(SEED)

        # Group indices by category
        self.cat_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.valid_items):
            self.cat_to_indices[item['category']].append(idx)
            
        self.categories = list(self.cat_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each category
        for cat in self.categories:
            random.shuffle(self.cat_to_indices[cat])
            
        # Track our position in each category list
        pointers = {cat: 0 for cat in self.categories}
        
        total_batches = len(self.dataset) // self.batch_size
        
        for _ in range(total_batches):
            batch = []
            # Round-robin category selection to guarantee diverse batches
            cat_idx = 0
            while len(batch) < self.batch_size:
                cat = self.categories[cat_idx % len(self.categories)]
                
                # If we haven't exhausted this category, pull an index
                if pointers[cat] < len(self.cat_to_indices[cat]):
                    idx = self.cat_to_indices[cat][pointers[cat]]
                    batch.append(idx)
                    pointers[cat] += 1
                    
                cat_idx += 1
                
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


if __name__ == "__main__":
    print("--- Running Triplet Dataset & Sampler Standalone Test ---")
    
    # We use a batch size of 4 to stick to the 6GB VRAM limit during actual training
    BATCH_SIZE = 4
    
    processed_dir = PROJECT_ROOT / "data" / "processed"
    renders_dir = PROJECT_ROOT / "data" / "renders"
    
    print(f"Loading dataset from: {processed_dir}")
    dataset = Pix3DContrastiveDataset(PIX3D_JSON, processed_dir, renders_dir)
    print(f"Total valid 2D-3D pairs found: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset is empty. Ensure preprocessing and rendering scripts have been run on 'data/' directories.")
        sys.exit(1)
        
    sampler = StratifiedBatchSampler(dataset, batch_size=BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    
    # Fetch a single batch to verify shapes and stratification
    batch_2d, batch_3d, batch_cats = next(iter(dataloader))
    
    print("\nBatch successfully loaded.")
    print(f"2D Queries Shape: {batch_2d.shape}")
    print(f"3D Renders Shape: {batch_3d.shape}")
    print(f"Categories in Batch: {batch_cats}")
    
    assert batch_2d.shape == (BATCH_SIZE, 3, 224, 224), "2D batch shape mismatch."
    assert batch_3d.shape == (BATCH_SIZE, 12, 3, 224, 224), "3D batch shape mismatch."
    assert len(set(batch_cats)) > 1, "Sampler failed to diversify categories. Batch contains only one class."
    
    print("Tensor shapes and category stratification verified.")
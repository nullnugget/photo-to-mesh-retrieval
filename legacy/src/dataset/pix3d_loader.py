"""
Dataset loaders for the Pix3D dataset.

Handles metadata parsing, deduplication of 3D CAD models, and path resolution 
for the offline rendering pipeline.
"""

import sys
import json
from pathlib import Path
from typing import Tuple
from collections import Counter

from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import PIX3D_JSON, PIX3D_DIR

class Pix3DUniqueModelsDataset(Dataset):
    """
    Parses Pix3D metadata to yield strictly unique 3D CAD models, bypassing the 
    many-to-one mapping of 2D images to 3D shapes.
    """
    
    def __init__(self, metadata_path: Path = PIX3D_JSON, root_dir: Path = PIX3D_DIR):
        self.root_dir = root_dir
        
        with open(metadata_path, 'r') as f:
            raw_metadata = json.load(f)
            
        # Deduplicate based on the 'model' relative path
        self.unique_models = []
        seen_models = set()
        
        for item in raw_metadata:
            model_rel_path = item['model']
            if model_rel_path not in seen_models:
                seen_models.add(model_rel_path)
                self.unique_models.append({
                    'category': item['category'],
                    'model_path': model_rel_path
                })
                
    def __len__(self) -> int:
        return len(self.unique_models)
        
    def __getitem__(self, idx: int) -> Tuple[Path, str, str]:
        item = self.unique_models[idx]
        
        category = item['category']
        rel_path = Path(item['model_path'])
        
        # Absolute path to the .obj file
        absolute_model_path = self.root_dir / rel_path
        
        # Extract the model identifier
        model_id = rel_path.parent.name
        
        return absolute_model_path, category, model_id

if __name__ == "__main__":
    print("--- Running Pix3D Loader Standalone Test ---")
    
    try:
        dataset = Pix3DUniqueModelsDataset()
        print(f"\n Total Unique 3D CAD Models: {len(dataset)}")
        
        counts = Counter(item['category'] for item in dataset.unique_models)
        
        print("\n Breakdown by Category:")
        for cat, count in counts.most_common():
            print(f"   - {cat.capitalize().ljust(10)} : {count}")
            
        if len(dataset) > 0:
            path, cat, mid = dataset[0]
            print(f"\n Sample output (Index 0):")
            print(f"   - Category : {cat}")
            print(f"   - Model ID : {mid}")
            print(f"   - Abs Path : {path}")
            
    except Exception as e:
        print(f"Error during loader test: {e}")
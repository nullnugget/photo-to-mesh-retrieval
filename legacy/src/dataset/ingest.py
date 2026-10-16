'''
This module's job is to parse the pix3d.json metadata 
and verify that the paths it points to actually exist in the local data/pix3d/ folder.
'''
import sys
import json
from pathlib import Path

# Add project root to path so we can import configs
sys.path.append(str(Path(__file__).resolve().parents[2]))

from configs.config import PIX3D_JSON, PIX3D_DIR

def load_pix3d_metadata(json_path: Path = PIX3D_JSON) -> list:
    """
    Loads and parses the pix3d.json metadata file.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {json_path}")
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def verify_dataset_integrity(metadata: list, num_samples: int = 5) -> bool:
    """
    Checks if the image and 3D model paths in the metadata actually exist on disk.
    Runs on a small subset (canary test).
    """
    print(f"Verifying {num_samples} samples from the dataset...")
    all_good = True
    
    for i in range(min(num_samples, len(metadata))):
        item = metadata[i]
        
        # Pix3D paths in the JSON are relative to the pix3d dataset root
        img_path = PIX3D_DIR / item['img']
        model_path = PIX3D_DIR / item['model']
        
        img_exists = img_path.exists()
        model_exists = model_path.exists()
        
        status = "OK" if (img_exists and model_exists) else "MISSING FILES"
        print(f"Sample {i+1}: Category [{item.get('category', 'unknown')}] -> {status}")
        
        if not img_exists:
            print(f"  -> Missing Image: {img_path}")
            all_good = False
        if not model_exists:
            print(f"  -> Missing Model: {model_path}")
            all_good = False
            
    return all_good

if __name__ == "__main__":
    print(f"--- Running Data Ingestion Test ---")
    try:
        data = load_pix3d_metadata()
        print(f"Successfully loaded metadata containing {len(data)} entries.")
        
        # Run a canary test to ensure paths resolve correctly
        is_intact = verify_dataset_integrity(data, num_samples=10)
        
        if is_intact:
            print("\nTest passed: Dataset paths are resolving correctly.")
        else:
            print("\nTest failed: Some files are missing.")
            
    except Exception as e:
        print(f"\nError during execution: {e}")
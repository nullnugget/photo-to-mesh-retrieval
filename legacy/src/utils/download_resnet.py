"""
Utility script to download the pre-trained ResNet50 model weights 
into the designated repository weights directory.
"""

import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import WEIGHTS_DIR

def main():
    resnet_dir = WEIGHTS_DIR / "ResNet50"
    resnet_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = resnet_dir / "resnet50.pth"
    url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    
    if target_path.exists():
        print(f"ResNet50 weights already exist at: {target_path}")
        return
        
    print(f"Downloading ResNet50 weights to {target_path}...")
    torch.hub.download_url_to_file(url, str(target_path))
    print("Download complete.")

if __name__ == "__main__":
    main()
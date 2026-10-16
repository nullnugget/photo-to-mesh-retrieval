import os
import torch
import random
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
# Resolves to /home/ami/workspace/xdim-cbir
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PIX3D_DIR = DATA_DIR / "pix3d"
PIX3D_JSON = PIX3D_DIR / "pix3d.json"

PROCESSED_DIR = DATA_DIR / "processed"
RENDERS_DIR = DATA_DIR / "renders"
INDICES_DIR = DATA_DIR / "indices"

# Weights path
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Force torchvision to download ResNet weights into the repo directory
os.environ["TORCH_HOME"] = str(WEIGHTS_DIR)

# -----------------------------------------------------------------------------
# Hardware & Reproducibility
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed: int = SEED):
    """Sets seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Pipeline Hyperparameters
# -----------------------------------------------------------------------------
# RTX 4050 6GB VRAM limits - conservative batch size for multi-view rendering
RENDER_BATCH_SIZE = 4 
IMG_SIZE = 224
NUM_VIEWS = 12
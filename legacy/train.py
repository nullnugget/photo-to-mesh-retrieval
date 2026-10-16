"""
Main training loop for the Siamese 2D-to-3D CBIR network.

Enforces a 6GB VRAM limit using FP16 mixed precision and gradient accumulation. 
Integrates a train/val split, logging, ETA tracking via tqdm, and best-model checkpointing.
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from configs.config import PROJECT_ROOT, PIX3D_JSON, DEVICE, SEED
from src.dataset.triplet_dataset import Pix3DContrastiveDataset, StratifiedBatchSampler
from src.models.encoders import SiameseCBIR
from src.models.loss import CrossModalInfoNCELoss

def setup_logger(log_path: Path) -> logging.Logger:
    """Configures dual logging to both the terminal and a persistent .log file."""
    logger = logging.getLogger("CBIR_Training")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    # Prevent duplicate handlers if the logger is initialized multiple times
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def get_train_val_split(metadata_path: Path, val_ratio: float = 0.15, is_mini: bool = False):
    """
    Splits the Pix3D metadata into distinct training and validation sets.
    If is_mini is True, truncates the dataset to 100 items for rapid pipeline testing.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    random.seed(SEED)
    random.shuffle(metadata)
    
    if is_mini:
        metadata = metadata[:100]
        
    split_idx = int(len(metadata) * (1.0 - val_ratio))
    return metadata[:split_idx], metadata[split_idx:]

def train(is_mini: bool = False):
    """
    Executes the main training and validation loop.
    
    Args:
        is_mini (bool): If True, runs a 1-epoch test on a micro-dataset and disables saving.
    """
    # --- Hyperparameters & VRAM Constraints ---
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 8  # Simulates an effective batch size of 32 (4 * 8)
    EPOCHS = 1 if is_mini else 30
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4         # Optimized for 16GB system RAM to prevent I/O bottlenecks
    
    # --- Directory Management ---
    processed_dir = PROJECT_ROOT / "data" / "processed"
    renders_dir = PROJECT_ROOT / "data" / "renders"
    checkpoints_dir = PROJECT_ROOT / "weights" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = checkpoints_dir / ("train_mini.log" if is_mini else "train.log")
    logger = setup_logger(log_file)
    mode_str = "MINI TEST MODE" if is_mini else "FULL PRODUCTION MODE"
    logger.info(f"--- Initializing Training Pipeline on {DEVICE} ({mode_str}) ---")
    
    # --- Dataset & DataLoader Initialization ---
    train_meta, val_meta = get_train_val_split(PIX3D_JSON, is_mini=is_mini)
    
    train_dataset = Pix3DContrastiveDataset(train_meta, processed_dir, renders_dir)
    val_dataset = Pix3DContrastiveDataset(val_meta, processed_dir, renders_dir)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Dataset partitioning failed or directories are empty.")
        sys.exit(1)
        
    # Apply category stratification to ensure diverse negative samples in every batch
    train_sampler = StratifiedBatchSampler(train_dataset, batch_size=BATCH_SIZE)
    val_sampler = StratifiedBatchSampler(val_dataset, batch_size=BATCH_SIZE)
    
    # pin_memory=True speeds up the host-to-device (CPU to GPU) tensor transfers
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    
    logger.info(f"Train Pairs: {len(train_dataset)} | Val Pairs: {len(val_dataset)}")
    
    # --- Model, Loss, & Optimizer ---
    model = SiameseCBIR().to(DEVICE)
    criterion = CrossModalInfoNCELoss(temperature=0.07).to(DEVICE)
    
    # AdamW includes weight decay regularization out of the box
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # GradScaler handles FP16 dynamic range to prevent gradient underflow
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    logger.info(f"Total Epochs: {EPOCHS} | Train Steps/Epoch: {len(train_loader)}")
    logger.info("Starting Training Loop...\n")
    
    for epoch in range(EPOCHS):
        # ==========================================
        #               TRAINING PHASE
        # ==========================================
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")
        
        for step, (batch_2d, batch_3d, _) in enumerate(train_pbar):
            # non_blocking=True allows overlapping data transfers with computation
            batch_2d = batch_2d.to(DEVICE, non_blocking=True)
            batch_3d = batch_3d.to(DEVICE, non_blocking=True)
            
            # 1. Forward Pass (Mixed Precision)
            with torch.amp.autocast('cuda'):
                embed_2d = model.forward_2d(batch_2d)
                embed_3d = model.forward_3d(batch_3d)
                
                loss = criterion(embed_2d, embed_3d)
                
                # Normalize loss mathematically before accumulating gradients
                loss = loss / ACCUMULATION_STEPS
                
            # 2. Backward Pass (Scaled for FP16)
            scaler.scale(loss).backward()
            
            # 3. Optimizer Step (Deferred for Gradient Accumulation)
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Aggressive cache clearing to prevent VRAM fragmentation
                torch.cuda.empty_cache()
                
            # 4. Metric Tracking (Revert normalization for accurate display)
            real_loss = loss.item() * ACCUMULATION_STEPS
            train_loss += real_loss
            train_pbar.set_postfix({'loss': f"{real_loss:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # ==========================================
        #              VALIDATION PHASE
        # ==========================================
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch", leave=False)
        
        with torch.no_grad():
            for batch_2d, batch_3d, _ in val_pbar:
                batch_2d = batch_2d.to(DEVICE, non_blocking=True)
                batch_3d = batch_3d.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    embed_2d = model.forward_2d(batch_2d)
                    embed_3d = model.forward_3d(batch_3d)
                    v_loss = criterion(embed_2d, embed_3d)
                    
                val_loss += v_loss.item()
                val_pbar.set_postfix({'val_loss': f"{v_loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # ==========================================
        #               CHECKPOINTING
        # ==========================================
        # Skip writing to disk during mini-tests to avoid overwriting production models
        if not is_mini:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), checkpoints_dir / "best_model.pth")
                logger.info(f"New best model saved! (Val Loss: {best_val_loss:.4f})")
                
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), checkpoints_dir / f"epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 2D-to-3D CBIR Siamese Network")
    parser.add_argument('--mini', action='store_true', help="Run a fast 1-epoch test on a tiny data subset")
    args = parser.parse_args()
    
    train(is_mini=args.mini)
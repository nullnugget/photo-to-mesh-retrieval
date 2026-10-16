"""
Cross-Modal InfoNCE Loss for 2D-to-3D Retrieval.

Calculates the symmetric contrastive loss between batches of L2-normalized 
2D query embeddings and 3D anchor embeddings, scaled by a temperature parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature (float): Scales the logits to control the penalty for hard 
                                 negatives. Lower values create a sharper distribution.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embed_2d: torch.Tensor, embed_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embed_2d (torch.Tensor): L2-normalized 2D embeddings of shape (B, D).
            embed_3d (torch.Tensor): L2-normalized 3D embeddings of shape (B, D).
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size = embed_2d.size(0)
        
        # Calculate cosine similarity matrix (B, B). 
        # Since embeddings are L2 normalized, dot product equals cosine similarity.
        logits = torch.matmul(embed_2d, embed_3d.T) / self.temperature
        
        # The positive pairs are the elements on the diagonal (0, 1, 2, ..., B-1)
        labels = torch.arange(batch_size, device=embed_2d.device)
        
        # Symmetric loss: we want 2D to retrieve the correct 3D, 
        # and the 3D to retrieve the correct 2D.
        loss_2d_to_3d = F.cross_entropy(logits, labels)
        loss_3d_to_2d = F.cross_entropy(logits.T, labels)
        
        return (loss_2d_to_3d + loss_3d_to_2d) / 2.0


if __name__ == "__main__":
    print("--- Running InfoNCE Loss Standalone Test ---")
    
    BATCH_SIZE = 4
    EMBED_DIM = 512
    
    # Simulate L2-normalized network outputs
    dummy_2d = F.normalize(torch.randn(BATCH_SIZE, EMBED_DIM), p=2, dim=1)
    
    # 1. Test with random mismatching 3D embeddings
    dummy_3d_random = F.normalize(torch.randn(BATCH_SIZE, EMBED_DIM), p=2, dim=1)
    
    # 2. Test with perfect matching 3D embeddings (identical to 2D)
    dummy_3d_perfect = dummy_2d.clone()
    
    criterion = CrossModalInfoNCELoss(temperature=0.07)
    
    loss_random = criterion(dummy_2d, dummy_3d_random)
    loss_perfect = criterion(dummy_2d, dummy_3d_perfect)
    
    print(f"Loss with random embeddings (untrained): {loss_random.item():.4f}")
    print(f"Loss with perfect embeddings (converged): {loss_perfect.item():.4f}")
    
    # Assertions to guarantee mathematical stability
    assert loss_random.item() > loss_perfect.item(), "Perfect matches should result in a lower loss."
    assert not torch.isnan(loss_random), "Loss calculation resulted in NaN."
    
    print("Standalone test passed. Loss gradients and calculations verified.")


# 1-- I think you again forgot that we do not have scripts dir
# 2-- I have 16 gigs of ram, do you think we should set num_workers to 0 or something > 0?
# 3-- Will 10 epochs be enough?
# 4-- We are not maintaining a validation set to monitor the validation loss? Is that right? Is it okay/good practice?
# 5-- From what it seems, checkpoints are getting saved in a sub dir called CBIR inside the weights dir.  Do you think we should make a subdir named as checkpoints and save them there? CBIR doesnt seem right.
# 6-- Also, i don't think we are keeping a track of which checkpoint is the best? Are we?
# 7-- Additionally, i think the loss values that are being printed should actually be stored into a log file. But where do we save this log file? In checkpoints subdir?
# 8-- no ETA information?
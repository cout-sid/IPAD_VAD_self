import torch
import torch.nn as nn

class MotionAttentionMask(nn.Module):
    """
    Learns a soft spatial mask from temporal motion in the input clip.
    
    Input:  (B, 3, T, H, W) — raw video clip in [-1, 1]
    Output: (B, 1, 1, H, W) — soft mask in [base_weight, 1.0]
    
    The mask is high where motion occurs, low on static background.
    base_weight is learnable — no manual tuning needed.
    """
    
    def __init__(self):
        super().__init__()
        
        # Learnable base weight for static regions (initialized at 0.3)
        # sigmoid ensures it stays in (0, 1)
        self.base_weight_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5, will learn
        
        # Small conv network to refine raw motion into a smooth mask
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output in [0, 1]
        )
    
    def forward(self, clip):
        """
        Args:
            clip: (B, 3, T, H, W)
        Returns:
            mask: (B, 1, 1, H, W) — weight map for loss
        """
        B, C, T, H, W = clip.shape
        
        # 1. Compute motion: mean absolute temporal difference
        clip_01 = (clip + 1) / 2.0  # [-1,1] → [0,1]
        
        diffs = []
        for t in range(1, T):
            diffs.append(torch.abs(clip_01[:, :, t] - clip_01[:, :, t-1]))
        
        # Average across all frame pairs and channels → (B, 1, H, W)
        motion = torch.stack(diffs, dim=0).mean(dim=0).mean(dim=1, keepdim=True)
        
        # 2. Refine motion into smooth mask → (B, 1, H, W)
        attention = self.refine(motion)  # [0, 1]
        
        # 3. Scale to [base_weight, 1.0]
        base_weight = torch.sigmoid(self.base_weight_logit)  # learnable, in (0, 1)
        mask = base_weight + (1.0 - base_weight) * attention
        
        # 4. Reshape for broadcasting with loss: (B, 1, 1, H, W)
        mask = mask.unsqueeze(2)
        
        return mask, base_weight
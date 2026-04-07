from pytorch_wavelets import DWTForward, DWTInverse
import torch
import torch.nn as nn
from torch.nn import functional as F

class DWTFeatureGate(nn.Module):
    """
    Applies DWT along spatial dims of 3D feature maps.
    Suppresses LL (background/smooth) and retains HF (texture/edges).
    Input/output shape: (B, C, D, H, W) — unchanged.
    """
    def __init__(self, channels, wave='db1'):
        super().__init__()
        self.dwt  = DWTForward(J=1, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')
        
        # learnable gate: how much LL to suppress (per channel)
        self.ll_gate = nn.Parameter(torch.ones(1, channels, 1, 1, 1) * 0.3)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        frames_out = []
        
        for t in range(D):
            frame = x[:, :, t]           # (B, C, H, W)
            yl, yh = self.dwt(frame)     # yl: (B,C,H/2,W/2), yh: high-freq bands
            
            # suppress LL — gate controls how much background leaks through
            yl_gated = yl * self.ll_gate.squeeze(-1).squeeze(-1).squeeze(-1).view(1, C, 1, 1)
            
            frame_out = self.idwt((yl_gated, yh))
            frame_out = frame_out[:, :, :H, :W]
            frames_out.append(frame_out)
        
        return torch.stack(frames_out, dim=2)  # (B, C, D, H, W)
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """
    Simple Haar Wavelet 2D Decomposition
    Applies per-channel wavelet decomposition
    """
    def __init__(self):
        super(HaarDWT2D, self).__init__()

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        Returns:
            LL, LH, HL, HH  (same shape except spatial halved)
        """

        B, C, T, H, W = x.shape

        # Reshape to merge temporal into batch
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B*T, C, H, W)

        # Haar filters
        LL = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] +
              x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4

        LH = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] +
              x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 4

        HL = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] -
              x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 4

        HH = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] -
              x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4

        # reshape back
        LL = LL.reshape(B, T, C, H//2, W//2).permute(0, 2, 1, 3, 4)
        LH = LH.reshape(B, T, C, H//2, W//2).permute(0, 2, 1, 3, 4)
        HL = HL.reshape(B, T, C, H//2, W//2).permute(0, 2, 1, 3, 4)
        HH = HH.reshape(B, T, C, H//2, W//2).permute(0, 2, 1, 3, 4)

        return LL, LH, HL, HH

class WaveletAttention(nn.Module):
    """
    Wavelet-based High Frequency Attention with Channel Bottleneck and Residual Connection
    """
    def __init__(self, channels=768, reduced_channels=32):
        super(WaveletAttention, self).__init__()

        self.dwt = HaarDWT2D()
        
        # Step 1: Bottleneck - Reduce channels to extract core spatial features
        self.reduce = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        
        # Step 2: High-frequency processing
        # Processes the high-frequency energy map
        self.conv_hf = nn.Sequential(
            nn.Conv3d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # Step 3: Expansion - Project back to 768 channels to create the attention mask
        self.expand = nn.Conv3d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, T, H, W) where C=768, H=8, W=8
        """
        B, C, T, H, W = x.shape

        # 1. Channel Reduction
        x_red = self.reduce(x) # (B, reduced_channels, T, 8, 8)

        # 2. DWT on reduced features
        LL, LH, HL, HH = self.dwt(x_red) # Spatial resolution becomes 4x4

        # 3. Calculate High-Frequency Energy
        hf_energy = torch.abs(LH) + torch.abs(HL) + torch.abs(HH)
        
        # 4. Refine HF map (Optional but helps smoothing)
        hf_energy = self.conv_hf(hf_energy)

        # 5. Upsample back to 8x8
        # We handle T separately to use 2D-based interpolation efficiently if needed,
        # but 3D interpolate works fine for spatial-only scaling.
        hf_upsampled = F.interpolate(
            hf_energy,
            size=(T, H, W), # (T, 8, 8)
            mode='trilinear',
            align_corners=False
        )

        # 6. Generate Attention Mask (Expand back to 768 channels)
        att = self.sigmoid(self.expand(hf_upsampled))

        # 7. Residual Connection: Original + (Original * Attention)
        # This ensures that even if attention is 0, the base features remain.
        out = x + (x * att) 

        return out, att

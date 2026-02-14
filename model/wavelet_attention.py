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
    Wavelet-based High Frequency Attention
    """
    def __init__(self, channels):
        super(WaveletAttention, self).__init__()

        self.dwt = HaarDWT2D()

        # Reduce high-frequency maps back to original size
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """

        LL, LH, HL, HH = self.dwt(x)

        # High-frequency energy
        hf_energy = torch.abs(LH) + torch.abs(HL) + torch.abs(HH)

        B, C, T, H, W = hf_energy.shape
        hf_energy = hf_energy.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        hf_energy = hf_energy.reshape(B*T, C, H, W)


        # Upsample back to original size
        hf_energy = F.interpolate(
            hf_energy,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # Restore temporal dimension shape
        hf_energy = hf_energy.reshape(B, T, C, x.shape[-2], x.shape[-1])
        hf_energy = hf_energy.permute(0, 2, 1, 3, 4)




        # Generate attention
        att = self.sigmoid(self.conv(hf_energy))

        # Reweight original feature
        out = x * att

        return out, att

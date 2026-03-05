import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse  # Requires pip install pytorch_wavelets

class AdvancedWaveletAttention(nn.Module):
    def __init__(self, channels, wavelet='db4'):
        super(AdvancedWaveletAttention, self).__init__()
        # Use J=1 for single level decomposition; wave='db4' is smoother than Haar
        self.xfm = DWTForward(J=1, wave=wavelet, mode='reflect')
        self.ifm = DWTInverse(wave=wavelet, mode='reflect')
        
        # Channel attention to weight the importance of sub-bands
        # DWT produces 4 sub-bands (LL, LH, HL, HH); we weight them individually
        self.subband_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        
        # We apply 2D DWT per temporal slice (T) or treat T as batch
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        
        # Forward DWT: Yl (low freq), Yh (list of high freq details)
        Yl, Yh = self.xfm(x_reshaped)
        
        # Process high-frequency details (Yh[0] is shape (B*T, C, 3, H/2, W/2))
        # We can apply attention here to emphasize specific detail orientations
        details = Yh[0]
        # Example: Simple spatial-frequency attention
        att_mask = torch.sigmoid(details)
        enhanced_details = details * att_mask
        
        # Inverse DWT to reconstruct the enhanced feature map
        out_reshaped = self.ifm((Yl, [enhanced_details]))
        
        # Restore original shape
        out = out_reshaped.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        
        # Final channel-wise calibration
        return out * self.subband_attention(out)
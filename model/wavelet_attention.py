import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class AdvancedWaveletAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.dwt = DWTForward(J=1, wave='haar', mode='reflect')

        self.channel_reduce = nn.Conv3d(
            channels * 4,
            channels,
            kernel_size=1
        )

        # restore spatial resolution
        self.upsample = nn.Upsample(
            scale_factor=(1,2,2),
            mode='trilinear',
            align_corners=False
        )

    def forward(self, x):

        B,C,T,H,W = x.shape

        x = x.permute(0,2,1,3,4).reshape(-1,C,H,W)

        Yl, Yh = self.dwt(x)

        details = Yh[0]

        LH = details[:,:,0,:,:]
        HL = details[:,:,1,:,:]
        HH = details[:,:,2,:,:]

        wavelet_features = torch.cat([Yl, LH, HL, HH], dim=1)

        wavelet_features = wavelet_features.view(
            B,T,4*C,H//2,W//2
        ).permute(0,2,1,3,4)

        out = self.channel_reduce(wavelet_features)

        # restore 8x8 resolution
        out = self.upsample(out)

        return out
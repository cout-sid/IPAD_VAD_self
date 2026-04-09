
from functools import reduce
from operator import mul



from pytorch_wavelets import DWTForward, DWTInverse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.fft

class Reconstruction3DEncoder(nn.Module):
    def __init__(self, chnum_in):
        super(Reconstruction3DEncoder, self).__init__()

        # Dong Gong's paper code
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Reconstruction3DDecoder(nn.Module):
    def __init__(self, chnum_in):
        super(Reconstruction3DDecoder, self).__init__()

        # Dong Gong's paper code + Tanh
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# class VST3DDecoder(nn.Module):
#     def __init__(self, chnum_out):
#         super(VST3DDecoder, self).__init__()

#         # Dong Gong's paper code + Tanh
#         self.chnum_out = chnum_out
#         feature_num = 128    # prev 128
#         feature_num_2 = 96   # prev 96
#         feature_num_x2 = 256  # prev 256
#         feature_num_in = 768
        
#         self.transformer_decoder = nn.Sequential(
            
#             # (768,2,8,8)
#             nn.ConvTranspose3d(feature_num_in, 512, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
#                                output_padding=(1, 1, 1)),
#             nn.BatchNorm3d(feature_num_x2),
#             nn.LeakyReLU(0.2, inplace=True),
#                 # 256,4,16,16
#             nn.ConvTranspose3d(512, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
#                                output_padding=(1, 1, 1)),
#             nn.BatchNorm3d(feature_num_x2),
#             nn.LeakyReLU(0.2, inplace=True),
#                 # 256,8,32,32

#             nn.ConvTranspose3d(feature_num_x2, feature_num, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
#                                output_padding=(0, 1, 1)),
#             nn.BatchNorm3d(feature_num),
#             nn.LeakyReLU(0.2, inplace=True),
#                 # 128,8,64,64

#             nn.ConvTranspose3d(feature_num, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
#                                output_padding=(0, 1, 1)),
#             nn.BatchNorm3d(feature_num_2),
#             nn.LeakyReLU(0.2, inplace=True),
                
#             # 96,8,128,128
#             nn.ConvTranspose3d(feature_num_2, self.chnum_out, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
#                                output_padding=(0, 1, 1)),

                
#             # 3,8,256,256
#             nn.ConvTranspose3d(self.chnum_out, self.chnum_out, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
#                                output_padding=(0, 0, 0)),
#             # 3,8,256,256
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.transformer_decoder(x)
#         return x

   


class VST3DDecoder(nn.Module):
    """
    Decoder for 8-frame input.
    Encoder output: (B, 768, 2, 8, 8) → target: (B, C, 8, 256, 256)

    Temporal:  2 → 4 → 8 → 8 → 8 → 8  (upsample first 2 stages, then hold)
    Spatial:   8 →16 →32 →64 →128→256  (upsample every stage)
    """

    def __init__(self, chnum_out, dropout=0.1):
        super().__init__()
        self.chnum_out = chnum_out

        # Stage 1: (768, 2, 8, 8) → (384, 4, 16, 16)  [temporal + spatial upsample]
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(768, 384, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
        )

        # Stage 2: (384, 4, 16, 16) → (256, 8, 32, 32)  [temporal + spatial upsample]
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(384, 256, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
        )

        # Stage 3: (256, 8, 32, 32) → (128, 8, 64, 64)  [spatial only]
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
        )

        # Stage 4: (128, 8, 64, 64) → (64, 8, 128, 128)  [spatial only + extra conv]
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 5: (64, 8, 128, 128) → (C, 8, 256, 256)  [spatial only + extra conv]
        self.up5 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, chnum_out, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (B, 768, 2, 8, 8)
        x = self.up1(x)   # (B, 384, 4, 16, 16)
        x = self.up2(x)   # (B, 256, 8, 32, 32)
        x = self.up3(x)   # (B, 128, 8, 64, 64)
        x = self.up4(x)   # (B,  64, 8, 128, 128)
        x = self.up5(x)   # (B,   C, 8, 256, 256)
        return x







import torch
import torch.nn as nn

class VSThalfDecoder(nn.Module):
    """
    For 128x128 input
    Input: (B, C, 2, 4, 4)
    Output: (B, chnum_out, T, 128, 128)
    """

    def __init__(self, in_channels=768, chnum_out=3, dropout=0.1):
        super().__init__()

        # 4 → 8
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 384, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
        )

        # 8 → 16
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(384, 192, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
        )

        # 16 → 32
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(192, 96, kernel_size=3,
                               stride=(1,2,2), padding=1,
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(96, 96, kernel_size=3, padding=1),   # 🔥 extra conv
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 32 → 64
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(96, 48, kernel_size=3,
                               stride=(1,2,2), padding=1,
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(48, 48, kernel_size=3, padding=1),   # 🔥 extra conv
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 64 → 128
        self.up5 = nn.Sequential(
            nn.ConvTranspose3d(48, 32, kernel_size=3,
                               stride=(1,2,2), padding=1,
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, chnum_out, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (B, C, 2, 4, 4)

        x = self.up1(x)   # (B, 384, 4, 8, 8)
        x = self.up2(x)   # (B, 192, 8, 16, 16)
        x = self.up3(x)   # (B, 96,  8, 32, 32)
        x = self.up4(x)   # (B, 48,  8, 64, 64)
        x = self.up5(x)   # (B, C,   8, 128, 128)

        return x




class DWTFeatureEnhance(nn.Module):
    """
    DWT sub-band convolution block for abstract feature maps.
    
    Instead of scalar-boosting high-freq bands (which amplifies noise
    equally with useful patterns), this decomposes into LL/LH/HL/HH,
    processes each band with its own depthwise-separable conv that can
    learn band-specific spatial transformations, then recombines via IDWT.
    
    The LL path is a simple identity (structure preservation).
    The high-freq paths learn to clean/sharpen within each orientation.
    A learnable gate per sub-band controls how much correction is applied.
    The whole block outputs a residual delta — so at init it's near-identity.
    
    Input/Output: (B, C, D, H, W) — shape unchanged.
    """
    def __init__(self, channels, wave='db4'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')
        
        # Per-sub-band depthwise-separable convs:
        # depthwise (spatial patterns within each channel) + pointwise (cross-channel mixing)
        # Lightweight: depthwise has C params per 3x3, pointwise has C*C params per 1x1
        self.conv_lh = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_hl = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_hh = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Per-sub-band gates: learnable scalars initialized near zero
        # so the block starts as near-identity and gradually learns corrections
        self.gate_lh = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate_hl = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate_hh = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        frames_out = []
        for t in range(D):
            frame = x[:, :, t, :, :]               # (B, C, H, W)
            
            yl, yh = self.dwt(frame)                # yl: (B,C,H',W')  yh: list
            lh = yh[0][:, :, 0]                     # (B, C, H'', W'')
            hl = yh[0][:, :, 1]
            hh = yh[0][:, :, 2]
            
            # Process each high-freq band with its own conv path
            # gate * conv(band) gives the learned correction; add to original band
            lh = lh + self.gate_lh * self.conv_lh(lh)
            hl = hl + self.gate_hl * self.conv_hl(hl)
            hh = hh + self.gate_hh * self.conv_hh(hh)
            
            # LL band passes through unchanged — preserve structure
            yh_proc = [torch.stack([lh, hl, hh], dim=2)]
            frame_out = self.idwt((yl, yh_proc))
            frame_out = frame_out[:, :, :H, :W]
            
            frames_out.append(frame_out)
        
        x_out = torch.stack(frames_out, dim=2)      # (B, C, D, H, W)
        return x_out

class FourierFeatureEnhance(nn.Module):
    """
    Fourier-based feature enhancement — comparison baseline against DWTFeatureEnhance.
    
    Applies FFT along spatial dims, learns to boost high-frequency components,
    then reconstructs via iFFT. 
    
    Input/Output: (B, C, D, H, W) — shape unchanged.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Predict high-freq boost from global average of feature map

        self.boost_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # (B, C, 1, 1)
            nn.Flatten(1),                         # (B, C)
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),   # one boost per channel
            nn.Softplus(),                         # boost > 0
        )

        # same fusion as DWTFeatureEnhance for fair comparison
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # threshold: frequency components above this (as fraction of max freq)
        # are considered "high frequency"
        self.hf_threshold = 0.4

    def forward(self, x):
        B, C, D, H, W = x.shape
        residual = x

        frames_out = []
        for t in range(D):
            frame = x[:, :, t, :, :]          # (B, C, H, W)

            # FFT — go to frequency domain (full complex spectrum)
            fft = torch.fft.fft2(frame, norm='ortho')          # (B, C, H, W) complex

            # separate magnitude and phase
            magnitude = fft.abs()              # (B, C, H, W)
            phase     = fft.angle()            # (B, C, H, W)

            # build high-freq mask
            freq_h = torch.fft.fftfreq(H, device=x.device)    # (H,)
            freq_w = torch.fft.fftfreq(W, device=x.device)    # (W,)
            freq_grid = (freq_h.unsqueeze(1) ** 2 +
                         freq_w.unsqueeze(0) ** 2).sqrt()      # (H, W)
            hf_mask = (freq_grid > self.hf_threshold).float()  # binary

            # predict boost from global average of magnitude (low-freq dominated)
            # mag_pooled = magnitude.mean(dim=[-2, -1], keepdim=True)  # (B, C, 1, 1)
            boosts = self.boost_predictor(magnitude)           # (B, C)
            boosts = boosts.view(B, C, 1, 1)                   # (B, C, 1, 1)

            # boost high-freq magnitude
            magnitude_boosted = magnitude * (1 + boosts * hf_mask)

            # reconstruct complex spectrum from boosted magnitude + original phase
            fft_boosted = torch.polar(magnitude_boosted, phase)

            # iFFT back to spatial domain, take real part to discard numerical noise
            frame_out = torch.fft.ifft2(fft_boosted, norm='ortho').real  # (B, C, H, W)
            frames_out.append(frame_out)

        x_out = torch.stack(frames_out, dim=2)   # (B, C, D, H, W)
        x_out = self.fusion(x_out) + residual
        return x_out

class DWTChannelAttention3D(nn.Module):
    """
    Process skip connections using only high-frequency DWT sub-bands.
    Discards LL (content) to prevent anomaly bypass through skips.
    Only LH, HL, HH (edges/textures) are kept and attended.
    """
    def __init__(self, channels, reduction=8, wave='db4'):
        super().__init__()
        self.channels = channels
        
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')
        
        # Attention only on high-freq bands (3 sub-bands = 3C)
        att_channels = channels * 3
        self.channel_att = nn.Sequential(
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(att_channels // reduction, att_channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        output_frames = []
        for t in range(D):
            frame = x[:, :, t, :, :]  # (B, C, H, W)
            
            yl, yh = self.dwt(frame)
            
            lh = yh[:, :, 0]
            hl = yh[:, :, 1]
            hh = yh[:, :, 2]
            
            # Only use high-freq bands
            high_freq = torch.cat([lh, hl, hh], dim=1)  # (B, 3C, H', W')
            
            # Channel attention on high-freq only
            att_weights = self.channel_att(high_freq)   # (B, 3C, H', W')
            high_freq = high_freq * att_weights
            
            lh_att, hl_att, hh_att = torch.chunk(high_freq, 3, dim=1)
            
            # Don'pass the low freq content to the model
            yl_zeros = torch.zeros_like(yl) 
            
            yh_att = [torch.stack([lh_att, hl_att, hh_att], dim=2)]
            
            # Reconstruct using zeros for LL, and attended features for high-freq
            frame_out = self.idwt((yl_zeros, yh_att))
            frame_out = frame_out[:, :, :H, :W]
            
            output_frames.append(frame_out)
        
        x_out = torch.stack(output_frames, dim=2)
        x_out = self.fusion(x_out) 
        
        return x_out


class VST3d_wavnet(nn.Module):
    """
    U-Net style decoder with DWT-processed skip connections from encoder.
    Symmetrical channel progression: 768 → 384 → 192 → 96 → 48 → 3
    
    Encoder skips used:
        skip[1]: (B, 384, T/4, 16, 16) → feeds into up1 output
        skip[0]: (B, 192, T/4, 32, 32) → feeds into up2 output
    
    Temporal alignment via trilinear interpolation (encoder T/4 → decoder T).
    use_skip=False disables skip connections for ablation.
    """

    def __init__(self, chnum_out, use_skip=False, dropout=0.1):
        super().__init__()
        self.chnum_out = chnum_out
        self.use_skip = use_skip

        # DWT enhancement at later decoder stages where spatial structure
        # is closer to the final output — sub-band convs can learn
        # meaningful orientation-specific patterns at these resolutions
        self.dwt_enhance_up3 = DWTFeatureEnhance(channels=96, wave='db4')   # after 64x64
        self.dwt_enhance_up4 = DWTFeatureEnhance(channels=48, wave='db4')   # after 128x128

        # Stage 1: (768, T/4, 8, 8) → (384, T/2, 16, 16)  [temporal + spatial upsample]
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(768, 384, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Skip connection processing for skip[1] (384-ch)
        if self.use_skip:
            self.dwt_att1 = DWTChannelAttention3D(channels=384, reduction=8, wave='db4')
            self.fuse1 = nn.Sequential(
                nn.Conv3d(384 + 384, 384, kernel_size=1),
                nn.BatchNorm3d(384),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Stage 2: (384, T/2, 16, 16) → (192, T, 32, 32)  [temporal + spatial upsample]
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(384, 192, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(192),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Skip connection processing for skip[0] (192-ch)
        if self.use_skip:
            self.dwt_att0 = DWTChannelAttention3D(channels=192, reduction=8, wave='db4')
            self.fuse0 = nn.Sequential(
                nn.Conv3d(192 + 192, 192, kernel_size=1),
                nn.BatchNorm3d(192),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Stage 3: (192, T, 32, 32) → (96, T, 64, 64)  [spatial only]
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(192, 96, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 4: (96, T, 64, 64) → (48, T, 128, 128)  [spatial only]
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(96, 48, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 5: (48, T, 128, 128) → (3, T, 256, 256)  [spatial only]
        self.up5 = nn.Sequential(
            nn.ConvTranspose3d(48, chnum_out, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.Tanh(),
        )

    def _temporal_align(self, skip, target_T):
        """Align skip temporal dim to target_T via trilinear interpolation."""
        _, _, D, H, W = skip.shape
        if D == target_T:
            return skip
        return F.interpolate(skip, size=(target_T, H, W), mode='trilinear', align_corners=False)

    def forward(self, x, skips=None):
        """
        Args:
            x: bottleneck feature (B, 768, T/4, 8, 8)
            skips: list of encoder skip features [skip0, skip1, skip2] or None
                   skip0: (B, 192, T/4, 32, 32)
                   skip1: (B, 384, T/4, 16, 16)
                   skip2: (B, 768, T/4, 8, 8)  — unused
        """
        # Stage 1: (768, T/4, 8, 8) → (384, T/2, 16, 16)
        x = self.up1(x)

        # Skip from encoder layer 1 (384-ch)
        if self.use_skip and skips is not None:
            s1 = self.dwt_att1(skips[1])                    # DWT attention
            s1 = self._temporal_align(s1, x.shape[2])       # temporal align
            x = self.fuse1(torch.cat([x, 0.5*s1], dim=1))       # concat + fuse

        # Stage 2: (384, T/2, 16, 16) → (192, T, 32, 32)
        x = self.up2(x)

        # Skip from encoder layer 0 (192-ch)
        if self.use_skip and skips is not None:
            s0 = self.dwt_att0(skips[0])                    # DWT attention
            s0 = self._temporal_align(s0, x.shape[2])       # temporal align
            x = self.fuse0(torch.cat([x, 0.5 * s0], dim=1))       # concat + fuse

        # Stage 3-5: spatial-only upsampling with DWT enhancement
        x = self.up3(x)                # (B,  96, T, 64, 64)
        # x = self.dwt_enhance_up3(x)    # sub-band conv refinement at 64x64
        x = self.up4(x)                # (B,  48, T, 128, 128)
        # x = self.dwt_enhance_up4(x)    # sub-band conv refinement at 128x128
        x = self.up5(x)                # (B,   3, T, 256, 256)
        return x
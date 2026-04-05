import torch.nn as nn
from functools import reduce
from operator import mul
import torch

from pytorch_wavelets import DWTForward, DWTInverse
import torch
import torch.nn as nn

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

    def __init__(self, chnum_out):
        super().__init__()
        self.chnum_out = chnum_out

        # Stage 1: (768, 2, 8, 8) → (384, 4, 16, 16)  [temporal + spatial upsample]
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(768, 384, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(384, 384, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 2: (384, 4, 16, 16) → (256, 8, 32, 32)  [temporal + spatial upsample]
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(384, 256, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 3: (256, 8, 32, 32) → (128, 8, 64, 64)  [spatial only]
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 4: (128, 8, 64, 64) → (64, 8, 128, 128)  [spatial only]
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 5: (64, 8, 128, 128) → (C, 8, 256, 256)  [spatial only]
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



# 8, 768, 4, 8, 8




class DWTChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=8, wave='db4'):
        super().__init__()
        self.channels = channels
        
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')
        
        # LL (C channels) + LH,HL,HH (3*C channels) = 4C
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels * 4, channels * 4 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 4 // reduction, channels * 4, bias=False),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        residual = x
        
        output_frames = []
        for t in range(D):
            frame = x[:, :, t, :, :]  # (B, C, H, W)
            
            # DWT: yl = (B, C, H/2, W/2), yh = list of (B, C, 3, H/2, W/2)
            yl, yh = self.dwt(frame)
            
            # yh[0] shape: (B, C, 3, H', W') — 3 sub-bands (LH, HL, HH)
            lh = yh[0][:, :, 0]  # (B, C, H', W')
            hl = yh[0][:, :, 1]
            hh = yh[0][:, :, 2]
            
            # Concatenate all sub-bands: (B, 4C, H', W')
            coeffs = torch.cat([yl, lh, hl, hh], dim=1)
            
            # Channel attention
            att_weights = self.channel_att(coeffs).view(B, 4 * C, 1, 1)
            coeffs = coeffs * att_weights
            
            # Split back
            yl_att, lh_att, hl_att, hh_att = torch.chunk(coeffs, 4, dim=1)
            yh_att = [torch.stack([lh_att, hl_att, hh_att], dim=2)]
            
            # IDWT
            frame_out = self.idwt((yl_att, yh_att))
            
            # Handle size mismatch from DWT padding
            frame_out = frame_out[:, :, :H, :W]
            
            output_frames.append(frame_out)
        
        x_out = torch.stack(output_frames, dim=2)  # (B, C, D, H, W)
        x_out = self.fusion(x_out) + residual
        
        return x_out

class VST3DDecoder_sixteen(nn.Module):
    """
    Decoder for 8-frame input.
    Encoder output: (B, 768, 4, 8, 8) → target: (B, C,16, 256, 256)

    Temporal:  4 → 8 → 16  (upsample first 2 stages, then hold)
    Spatial:   8 →16 →32 →64 →128→256  (upsample every stage)
    """

    def __init__(self, chnum_out, dropout=0.1):
        super().__init__()
        self.chnum_out = chnum_out

        # Stage 1: (768, 4, 8, 8) → (384, 8, 16, 16)  [temporal + spatial upsample]
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(768, 384, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(384),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout3d(dropout),
            # nn.Conv3d(384, 384, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 2: (384, 8, 16, 16) → (256, 8, 32, 32)  [temporal + spatial upsample]
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(384, 256, kernel_size=3,
                               stride=(2,2,2), padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout3d(dropout),
            # nn.Conv3d(256, 256, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 3: (256, 8, 32, 32) → (128, 8, 64, 64)  [spatial only]
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout3d(dropout),
            # nn.Conv3d(128, 128, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        # self.channel_att = ChannelAttention3D(channels=128, reduction=8)
        self.dwt_att = DWTChannelAttention3D(channels=128, reduction=8, wave='db4')


        # Stage 4: (128, 8, 64, 64) → (64, 8, 128, 128)  [spatial only]
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3),
                               stride=(1,2,2), padding=(1,1,1),
                               output_padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout3d(dropout),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 5: (64, 8, 128, 128) → (C, 8, 256, 256)  [spatial only]
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
        # x: (B, 768, 4, 8, 8)
        x = self.up1(x)   # (B, 384, 8, 16, 16)
        x = self.up2(x)   # (B, 256, 8, 32, 32)
        x = self.up3(x)   # (B, 128, 8, 64, 64)
        x = self.dwt_att(x)   # channel attention — same shape
        x = self.up4(x)   # (B,  64, 8, 128, 128)
        x = self.up5(x)   # (B,   C, 8, 256, 256)
        return x
    

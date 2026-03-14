import torch.nn as nn
from functools import reduce
from operator import mul
import torch

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
    def __init__(self, chnum_out):
        super().__init__()

        self.chnum_out = chnum_out

        # -------- Upsample stages --------
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(768, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(128, 96, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1)),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up5 = nn.ConvTranspose3d(
            96, chnum_out, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1)
        )

        # -------- Projection layers for residual alignment --------
        self.proj_x = nn.Conv3d(768, 256, 1)   # align x → up2
        self.proj_up2 = nn.Conv3d(256, 96, 1)  # align up2 → up4

        self.final = nn.Sequential(
            nn.Conv3d(chnum_out, chnum_out, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        x1 = self.up1(x)
        x2 = self.up2(x1)

        # residual: up2 + projected x
        x2 = x2 + self.proj_x(
            torch.nn.functional.interpolate(x, size=x2.shape[2:], mode="trilinear", align_corners=False)
        )

        x3 = self.up3(x2)
        x4 = self.up4(x3)

        # residual: up4 + projected up2
        x4 = x4 + self.proj_up2(
            torch.nn.functional.interpolate(x2, size=x4.shape[2:], mode="trilinear", align_corners=False)
        )

        x5 = self.up5(x4)

        return self.final(x5)
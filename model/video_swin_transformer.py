import torch
from .reconstruction_model import Reconstruction3DEncoder, Reconstruction3DDecoder, VST3DDecoder
from .VST_block import SwinTransformer3D
from einops import rearrange
from model import MemModule
import torch.nn as nn
from torch.nn import functional as F

from .wavelet_attention import AdvancedWaveletAttention # Updated from WaveletAttention


# from torch_vst_encoder import TorchVSTEncoder

# from video_swin_encoder import VideoSwinEncoder

# # Create model
# encoder = VideoSwinEncoder()
# encoder.eval()

# # Example input
# input_tensor = torch.randn(1, 3, 16, 224, 224)

# with torch.no_grad():
#     features = encoder(input_tensor)

# print("Feature shape:", features.shape)


class VST(torch.nn.Module):
    def __init__(self, mem_dim=2000, shrink_thres=0.0025):  # for reconstruction
        super(VST, self).__init__()
        self.reconstruction = True
        # self.chnum_in = chnum_in

        # self.encoder = Reconstruction3DEncoder(chnum_in=1)  # black and white
        # self.decoder = Reconstruction3DDecoder(chnum_in=1)  # black and white
        self.transformer_encoder = SwinTransformer3D()
        # self.transformer_encoder = TorchVSTEncoder()


        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=768, shrink_thres=shrink_thres)
        self.period = nn.Sequential(
            nn.Conv3d(768, 768, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(768),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size,768,4,4,4)
            nn.AdaptiveAvgPool3d((1, 1, 1)), 
            nn.Flatten(1),
            nn.Linear(768, 4096),
            # nn.Flatten(1),
            # nn.Linear(768*4*4*4,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,200),
        )
        self.transformer_decoder = VST3DDecoder(chnum_out=3)
        # self.encoder = Reconstruction3DEncoder(chnum_in=3)  # RGB
        # self.decoder = Reconstruction3DDecoder(chnum_in=3)  # RGB

        self.wavelet_att = AdvancedWaveletAttention(channels=768, wavelet='db4')


    def forward(self, x):
        
        feature = self.transformer_encoder(x)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # print("model debugging")
        # print(f"printing the shape of output of VST model {feature.shape}")
        #feature (batch_size,768,4,8,8)  --> previously now it's (batch_size,768,2,8,8)

        #wavelet transform
        feature = self.wavelet_att(feature)
        recon_index = self.period(feature)
        # print(f"The shape of recon_index i.e output of self.period: {recon_index.shape}") 
        # [8,200]
        # print(recon_index[0])
        res_mem = self.mem_rep(feature, recon_index)
        feature_mem = res_mem['output']
        # print(f"feature shape after memory module: {feature.shape}")
        # [8, 768, 2, 8, 8]
        att = res_mem['att']
        output = self.transformer_decoder(feature_mem.clone())
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        
        
        
        # print("The shape of output after decoding")
        # print(output.shape)
        return {'output': output, 'att': att, 'recon_index': recon_index}



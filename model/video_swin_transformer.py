import torch
from .reconstruction_model import Reconstruction3DEncoder, Reconstruction3DDecoder, VST3DDecoder, VST3d_wavnet
from .VST_block import SwinTransformer3D
from einops import rearrange
from model import MemModule
import torch.nn as nn
from torch.nn import functional as F

from .wavelet_attention import AdvancedWaveletAttention # Updated from WaveletAttention
from .motion_attention_mask import MotionAttentionMask


class VST(torch.nn.Module):
    def __init__(self, mem_dim=2000, shrink_thres=0.0025, use_skip=True):  # for reconstruction
        super(VST, self).__init__()
        self.reconstruction = True
        self.use_skip = use_skip

        self.transformer_encoder = SwinTransformer3D()

        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=768, shrink_thres=shrink_thres)
        self.period = nn.Sequential(
            nn.Conv3d(768, 768, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(768),
            nn.LeakyReLU(0.2, inplace=True),
            # (batch_size,768,4,4,4)
            nn.AdaptiveAvgPool3d((1, 1, 1)), 
            nn.Flatten(1),
            nn.Linear(768, 4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,200),
        )
        if use_skip:
            self.transformer_decoder = VST3d_wavnet(chnum_out=3, use_skip=use_skip)
        else:
            self.transformer_decoder=VST3DDecoder(chnum_out=3)

        self.wavelet_att = AdvancedWaveletAttention(channels=768)

    def forward(self, x):
        # Encoder: with or without skip connections
        if self.use_skip:
            feature, skips = self.transformer_encoder.forward_with_skips(x)
        else:
            feature = self.transformer_encoder(x)
            skips = None

        # Period prediction
        recon_index = self.period(feature)

        # Memory module
        res_mem = self.mem_rep(feature, recon_index)
        feature_mem = res_mem['output']
        att = res_mem['att']

        # Decoder: pass skips if available
        if self.use_skip:
            output = self.transformer_decoder(feature_mem.clone(), skips=skips)
        else:
            output = self.transformer_decoder(feature_mem.clone())
        return {
            'output': output,
            'att': att,
            'recon_index': recon_index,
        }

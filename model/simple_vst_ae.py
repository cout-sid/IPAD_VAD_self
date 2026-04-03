"""
SimpleVST_AE: A pure reconstruction model.

Encoder: Video Swin Transformer (from VST_block.py - SwinTransformer3D)
Decoder: VST3DDecoder_sixteen (3D ConvTranspose decoder)

No memory module, no period classifier, no entropy loss.
Just MSE reconstruction loss. The goal is to learn a general-purpose
video reconstructor that can reconstruct both normal and anomalous frames.

Input:  (B, 3, 16, 256, 256)
Encoder output: (B, 768, 4, 8, 8)
Decoder output: (B, 3, 16, 256, 256)
"""

import torch
import torch.nn as nn
from model.VST_block import SwinTransformer3D
from model.reconstruction_model import VST3DDecoder_sixteen


class SimpleVST_AE(nn.Module):
    def __init__(self):
        super(SimpleVST_AE, self).__init__()
        self.encoder = SwinTransformer3D()
        self.decoder = VST3DDecoder_sixteen(chnum_out=3)

    def forward(self, x):
        # x: (B, 3, 16, 256, 256)
        feature = self.encoder(x)
        # feature: (B, 768, 4, 8, 8)
        output = self.decoder(feature)
        # output: (B, 3, 16, 256, 256)
        return {'output': output}
# video_swin_encoder.py

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t


class TorchVSTEncoder(nn.Module):
    """
    Video Swin Transformer encoder WITHOUT pretrained weights.
    Outputs spatio-temporal feature maps (no classification).
    """

    def __init__(self):
        super().__init__()

        self.backbone = swin3d_t(weights=None)

        # Remove classification head completely
        self.backbone.head = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Tensor [B, 3, T, H, W]

        Returns:
            features: [B, T', H', W', C]  (e.g. [B, 2, 7, 7, 768])
        """
        x = self.backbone.features(x)
        x = self.backbone.norm(x)
        return x

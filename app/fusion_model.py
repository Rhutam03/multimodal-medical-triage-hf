import torch
import torch.nn as nn
from app.models.image_encoder import ImageEncoder
from app.models.text_encoder import TextEncoder
from app.models.fusion_head import FusionHead


class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_encoder = ImageEncoder(256)
        self.text_encoder = TextEncoder(256)
        self.fusion_head = FusionHead(512, num_classes)

    def forward(self, image=None, text=None):
        feats = []

        if image is not None:
            feats.append(self.image_encoder(image))

        if text is not None:
            feats.append(self.text_encoder(text))

        fused = torch.cat(feats, dim=1)
        return self.fusion_head(fused)

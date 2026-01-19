import torch
import torch.nn as nn

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_head import FusionHead


class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.fusion_head = FusionHead(num_classes)

    def forward(self, image, text):
        image_feat = self.image_encoder(image)
        text_feat = self.text_encoder(text)

        fused = torch.cat([image_feat, text_feat], dim=1)
        return self.fusion_head(fused)
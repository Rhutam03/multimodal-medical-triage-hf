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
        self.fusion_head = FusionHead(num_classes=num_classes)

    def forward(self, image=None, text=None):
        features = []

        if image is not None:
            img_feat = self.image_encoder(image)
            features.append(img_feat)

        if text is not None:
            txt_feat = self.text_encoder(text)
            features.append(txt_feat)

        fused = torch.cat(features, dim=1)
        return self.fusion_head(fused)

import torch
import torch.nn as nn

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_head import FusionHead


class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_encoder = ImageEncoder(output_dim=128)
        self.text_encoder = TextEncoder(output_dim=128)
        self.fusion_head = FusionHead(input_dim=256, num_classes=num_classes)

    def forward(self, image=None, text=None):
        features = []

        if image is not None:
            img_feat = self.image_encoder(image)
            if img_feat is not None:
                features.append(img_feat)

        if text is not None:
            txt_feat = self.text_encoder(text)
            if txt_feat is not None:
                features.append(txt_feat)

        if len(features) == 0:
            raise RuntimeError("Both image and text inputs are None")

        if len(features) == 1:
            fused = torch.cat([features[0], features[0]], dim=1)
        else:
            fused = torch.cat(features, dim=1)

        return self.fusion_head(fused)

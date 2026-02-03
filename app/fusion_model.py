import torch
import torch.nn as nn

from app.models.image_encoder import ImageEncoder
from app.models.text_encoder import TextEncoder


class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        fusion_dim = (
            self.image_encoder.output_dim
            + self.text_encoder.output_dim
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text):
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(text)

        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.fusion_head(fused)

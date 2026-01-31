import torch
import torch.nn as nn

from app.models.image_encoder import ImageEncoder
from app.models.text_encoder import TextEncoder


class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, text_feats):
        """
        images: [B, 3, 224, 224]
        text_feats: [B, 128]
        """

        img_feat = self.image_encoder(images)     # [B, 64]
        txt_feat = self.text_encoder(text_feats)  # [B, 64]

        # ✅ CORRECT CONCAT
        fused = torch.cat([img_feat, txt_feat], dim=1)  # [B, 128]

        return self.classifier(fused)

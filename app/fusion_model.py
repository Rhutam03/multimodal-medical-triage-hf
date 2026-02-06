import torch
import torch.nn as nn
from app.models.image_encoder import ImageEncoder
from app.models.text_encoder import TextEncoder

class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        fused_dim = self.image_encoder.output_dim + self.text_encoder.output_dim

        self.norm = nn.LayerNorm(fused_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, texts):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(texts)

        fused = torch.cat([img_feat, txt_feat], dim=1)
        fused = self.norm(fused)

        return self.classifier(fused)

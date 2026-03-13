import torch
import torch.nn as nn

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder


class MultimodalTriageModel(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 3):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(vocab_size=vocab_size)

        fusion_dim = self.image_encoder.out_dim + self.text_encoder.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, tokens):
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(tokens)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(fused)
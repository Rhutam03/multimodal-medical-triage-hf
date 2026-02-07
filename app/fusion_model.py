import torch
import torch.nn as nn
from app.models.image_encoder import ImageEncoder
from app.models.text_encoder import TextEncoder

class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, image, text_tokens):
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(text_tokens)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(fused)
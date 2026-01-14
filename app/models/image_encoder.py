import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, image_tensor):
        # image_tensor: (B, 3, 224, 224)
        features = self.encoder(image_tensor).squeeze(-1).squeeze(-1)
        return self.fc(features)

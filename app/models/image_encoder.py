import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    """
    Image encoder that accepts TENSOR images (B, 3, 224, 224)
    This is REQUIRED for training stability.
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove classifier
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: Tensor of shape (B, 3, 224, 224)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # safety

        features = self.encoder(image)          # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        return self.fc(features)                 # (B, output_dim)

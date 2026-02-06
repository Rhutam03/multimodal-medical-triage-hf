import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in backbone.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

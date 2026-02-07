import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()

        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        self.output_dim = 512

    def forward(self, x):
        return self.backbone(x)

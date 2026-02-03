import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(pretrained=True)

        # Remove classification head
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # ResNet18 outputs 512-dim features
        self.output_dim = 512

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 512]
        return x

import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)
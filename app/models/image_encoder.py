import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.encoder = nn.Sequential(
            *list(base.children())[:-1]
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

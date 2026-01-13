import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

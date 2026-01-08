import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(768 + 512, num_classes)

    def forward(self, x):
        return self.fc(x)

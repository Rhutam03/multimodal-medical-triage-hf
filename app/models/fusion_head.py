import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # low / medium / high
        )

    def forward(self, img_feat, txt_feat):
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(fused)

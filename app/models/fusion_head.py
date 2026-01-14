import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """
    Final classification head.
    Expects concatenated image + text features of size 512.
    Outputs triage logits: [low, medium, high].
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, fused_features: torch.Tensor):
        """
        Args:
            fused_features (Tensor): shape (512,) or (B, 512)
        Returns:
            logits (Tensor): shape (num_classes,) or (B, num_classes)
        """
        return self.classifier(fused_features)

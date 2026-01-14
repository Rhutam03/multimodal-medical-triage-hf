import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, text):
        """
        text: Tensor of shape (B, 128)
        """
        if text is None:
            return None

        if text.dim() == 1:
            text = text.unsqueeze(0)  # ensure batch dim

        return self.encoder(text)

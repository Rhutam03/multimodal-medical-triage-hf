import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, text: torch.Tensor):
        if text.dim() == 1:
            text = text.unsqueeze(0)
        return self.encoder(text)

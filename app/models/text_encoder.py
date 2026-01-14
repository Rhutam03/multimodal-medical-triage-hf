import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """
    Simple text encoder that accepts numeric text embeddings (Tensor).
    This matches training + inference pipelines cleanly.
    """

    def __init__(self, input_dim: int = 128, output_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, text: torch.Tensor):
        """
        Args:
            text (Tensor): shape (B, input_dim) or (input_dim,)

        Returns:
            Tensor: shape (B, output_dim)
        """
        if text.dim() == 1:
            text = text.unsqueeze(0)

        return self.encoder(text)

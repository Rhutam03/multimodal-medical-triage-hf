import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)
        self.output_dim = 128

    def forward(self, text):
        tokens = torch.randint(0, 10000, (text.shape[0], 10))
        return self.embedding(tokens).mean(dim=1)

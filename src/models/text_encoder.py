import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = embed_dim

    def forward(self, tokens):
        x = self.embedding(tokens)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x
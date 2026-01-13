import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, output_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if x is None:
            return None

        # x: (batch, seq_len)
        emb = self.embedding(x)          # (batch, seq_len, dim)
        emb = emb.permute(0, 2, 1)        # (batch, dim, seq_len)
        pooled = self.pool(emb).squeeze(-1)
        return pooled

import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)

    def forward(self, text_tokens):
        emb = self.embedding(text_tokens)
        return emb.mean(dim=1)
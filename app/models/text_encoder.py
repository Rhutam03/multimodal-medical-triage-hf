import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(10000, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def tokenize(self, texts):
        """
        Very simple tokenizer:
        - hashes words to integers
        - returns padded tensor [B, T]
        """
        tokenized = []
        for t in texts:
            tokens = [abs(hash(w)) % 10000 for w in t.lower().split()]
            tokenized.append(tokens)

        max_len = max(len(t) for t in tokenized)
        padded = [
            t + [0] * (max_len - len(t)) for t in tokenized
        ]

        return torch.tensor(padded, dtype=torch.long)

    def forward(self, text):
        """
        text: List[str] or Tuple[str]
        returns: [B, embed_dim]
        """
        if isinstance(text, (list, tuple)):
            tokens = self.tokenize(text).to(next(self.parameters()).device)
        else:
            raise TypeError("TextEncoder expects list/tuple of strings")

        embeds = self.embedding(tokens)          # [B, T, D]
        embeds = embeds.transpose(1, 2)          # [B, D, T]
        pooled = self.pool(embeds).squeeze(-1)   # [B, D]
        return pooled

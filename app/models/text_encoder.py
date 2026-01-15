import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        self.model = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )

        self.proj = nn.Linear(self.model.config.hidden_size, embed_dim)

    def forward(self, texts):
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls)

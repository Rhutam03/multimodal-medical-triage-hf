import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")

        for param in self.model.parameters():
            param.requires_grad = False

        self.output_dim = 768

    def forward(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(next(self.model.parameters()).device)

        outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :]

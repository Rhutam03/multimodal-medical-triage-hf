import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # 🔑 REQUIRED BY fusion_model.py
        self.output_dim = self.model.config.hidden_size  # 768

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}

        outputs = self.model(**encoded)

        # CLS token embedding
        return outputs.last_hidden_state[:, 0, :]

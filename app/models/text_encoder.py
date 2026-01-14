import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Encodes clinical text into a fixed-size embedding.
    Output dimension = 256
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", output_dim: int = 256):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)

    def forward(self, text: str):
        """
        Args:
            text (str): clinical description
        Returns:
            Tensor: shape (256,)
        """
        if text is None or text.strip() == "":
            return None

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            pooled = outputs.last_hidden_state[:, 0]  # CLS token

        return self.projection(pooled).squeeze(0)

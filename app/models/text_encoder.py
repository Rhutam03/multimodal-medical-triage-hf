import torch.nn as nn
from transformers import DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # Freeze BERT to reduce memory
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return output.last_hidden_state[:, 0, :]

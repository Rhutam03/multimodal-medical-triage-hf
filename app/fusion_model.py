import torch
import torch.nn as nn
from .models.image_encoder import ImageEncoder
from .models.text_encoder import TextEncoder
from .models.fusion_head import FusionHead

class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.classifier = FusionHead(num_classes)

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(fused)

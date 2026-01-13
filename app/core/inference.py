import torch
from app.models.image_encoder import ImageEncoder
from app.models.text_encoder import TextEncoder
from app.models.fusion_head import FusionHead

image_encoder = ImageEncoder()
text_encoder = TextEncoder()
fusion_head = FusionHead()

LABELS = ["Low Risk", "Medium Risk", "High Risk"]

def predict(image, text):
    img_feat = torch.zeros(1, 256)
    txt_feat = torch.zeros(1, 768)

    if image is not None:
        img_feat = image_encoder(image).unsqueeze(0)

    if text and text.strip():
        txt_feat = text_encoder.encode(text)

    logits = fusion_head(img_feat, txt_feat)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs).item()

    return f"Triage Level: {LABELS[pred]} (confidence {probs[0][pred]:.2f})"

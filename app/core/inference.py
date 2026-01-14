import torch
from app.fusion_model import MultimodalTriageModel

LABELS = ["Low", "Medium", "High"]

def load_model(weights_path: str):
    model = MultimodalTriageModel(num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def predict(model, image=None, text=None):
    with torch.no_grad():
        logits = model(image=image, text=text)
        pred = torch.argmax(logits, dim=1).item()
        return LABELS[pred]

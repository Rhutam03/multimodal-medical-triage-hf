import torch
from fusion_model import MultimodalTriageModel

LABELS = ["Low", "Medium", "High"]

def load_model(weights_path: str):
    model = MultimodalTriageModel(num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def predict_from_inputs(model, image=None, text=None):
    with torch.no_grad():
        logits = model(image=image, text=text)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()
    return f"{LABELS[pred]} (confidence: {confidence:.2f})"

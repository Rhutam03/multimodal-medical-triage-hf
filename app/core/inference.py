import torch
from app.fusion_model import MultimodalTriageModel

LABELS = ["Low", "Medium", "High"]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

_model = None


def load_model():
    global _model
    if _model is None:
        model = MultimodalTriageModel(num_classes=3)
        state = torch.load(
            "app/weights/model_weights.pth",
            map_location=DEVICE
        )
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


@torch.no_grad()
def predict_from_inputs(image=None, text=None):
    model = load_model()

    if image is None and (text is None or text.strip() == ""):
        return "❌ Provide image and/or text"

    if image is not None:
        image = image.to(DEVICE).unsqueeze(0)

    if text is not None:
        text = [text]  # MUST be List[str]

    logits = model(image=image, text=text)
    probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()

    return f"{LABELS[pred]} (confidence: {conf:.2f})"

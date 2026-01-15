import torch

LABELS = ["Low", "Medium", "High"]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

_model = None


def load_model():
    global _model
    if _model is None:
        from app.fusion_model import MultimodalTriageModel  # LAZY IMPORT

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

    if image is not None:
        image = image.to(DEVICE)

    logits = model(image=image, text=text)
    probs = torch.softmax(logits, dim=1)[0]

    idx = int(probs.argmax())
    confidence = float(probs[idx])

    return f"{LABELS[idx]} (confidence: {confidence:.2f})"

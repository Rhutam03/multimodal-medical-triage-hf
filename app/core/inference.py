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

    if image is None:
        return "Error: No image provided"

    if text is None or text.strip() == "":
        text = "No clinical description provided."

    logits = model(
        image=image,
        text=[text]   # ✅ CRITICAL FIX
    )

    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()

    return f"{LABELS[idx]} (confidence: {probs[idx]:.2f})"


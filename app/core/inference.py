import torch

LABELS = ["Low", "Medium", "High"]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

_model = None


def load_model():
    """
    Lazy-load the model to avoid circular imports.
    This is REQUIRED for Hugging Face Spaces.
    """
    global _model

    if _model is None:
        from fusion_model import MultimodalTriageModel  # ✅ lazy import

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

    if image is None and not text:
        return "Please provide an image or text."

    # image: PIL.Image or None
    # text: str or None
    logits = model(image=image, text=text)

    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()

    return f"{LABELS[idx]} (confidence: {probs[idx]:.2f})"

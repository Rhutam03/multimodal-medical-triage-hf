import torch

from fusion_model import MultimodalTriageModel
from preprocessing.image_preprocess import preprocess_image
from preprocessing.text_preprocess import preprocess_text

LABELS = ["Low", "Medium", "High"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

WEIGHTS_PATH = "app/weights/model_weights.pth"

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    model = MultimodalTriageModel(num_classes=len(LABELS))
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    _model = model
    return model


@torch.no_grad()
def predict_from_inputs(image=None, text=None):
    if image is None:
        return "Error: No image provided"

    model = load_model()

    image_tensor = preprocess_image(image)
    text_batch = preprocess_text(text)

    image_tensor = image_tensor.to(DEVICE)

    logits = model(
        image=image_tensor,
        text=text_batch
    )

    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()

    return f"{LABELS[idx]} (confidence: {probs[idx]:.2f})"

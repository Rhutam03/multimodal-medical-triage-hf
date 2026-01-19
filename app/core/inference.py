import torch

from app.fusion_model import MultimodalTriageModel
from app.preprocessing.image_preprocess import preprocess_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["Low", "Medium", "High"]
_model = None


def load_model():
    global _model
    if _model is None:
        _model = MultimodalTriageModel(num_classes=len(LABELS))
        _model.load_state_dict(
            torch.load("app/weights/model_weights.pth", map_location=DEVICE)
        )
        _model.to(DEVICE)
        _model.eval()
    return _model


@torch.no_grad()
def predict_from_inputs(image=None, text=None):
    # --- Image ---
    image_tensor = preprocess_image(image)
    if image_tensor is None:
        return "Error: Image preprocessing failed"

    image_tensor = image_tensor.to(DEVICE)

    # --- Text (MUST be list[str]) ---
    if text is None or text.strip() == "":
        text_batch = ["No clinical description provided."]
    else:
        text_batch = [text]

    model = load_model()
    logits = model(image=image_tensor, text=text_batch)

    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()

    return f"{LABELS[idx]} (confidence: {probs[idx]:.2f})"

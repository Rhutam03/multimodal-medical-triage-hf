import os
import torch

from fusion_model import MultimodalTriageModel
from preprocessing.image_preprocess import preprocess_image
from preprocessing.text_preprocess import preprocess_text

DEVICE = torch.device("cpu")
LABELS = ["Low Risk", "Medium Risk", "High Risk"]

_model = None


def get_model():
    global _model

    if _model is not None:
        return _model

    model = MultimodalTriageModel(num_classes=3)
    model.to(DEVICE)
    model.eval()

    weights_path = os.path.join("weights", "model.pt")

    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
        try:
            model.load_state_dict(
                torch.load(weights_path, map_location=DEVICE)
            )
            print("✅ Model weights loaded")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}")
            print("⚠️ Using random initialization")

    _model = model
    return _model


def predict(image, text):
    model = get_model()

    image_tensor = preprocess_image(image)
    text_tensor = preprocess_text(text)

    with torch.no_grad():
        outputs = model(image=image_tensor, text=text_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1).item()

    return LABELS[pred]

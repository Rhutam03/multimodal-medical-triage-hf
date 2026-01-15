import torch
import torchvision.transforms as T
from app.fusion_model import MultimodalTriageModel

LABELS = ["Low", "Medium", "High"]
DEVICE = torch.device("cpu")

_model = None

# Image preprocessing (MATCH training)
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
    if image is None:
        return "Error: No image provided"

    if text is None or text.strip() == "":
        text = "No clinical description provided."

    model = load_model()

    # ✅ PIL → Tensor
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    # ✅ Forward pass
    logits = model(
        image=image_tensor,
        text=[text]
    )

    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()

    return f"{LABELS[idx]} (confidence: {probs[idx]:.2f})"

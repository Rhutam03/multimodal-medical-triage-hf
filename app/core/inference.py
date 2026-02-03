import torch
from app.fusion_model import MultimodalTriageModel

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

model = MultimodalTriageModel(num_classes=3).to(DEVICE)
model.load_state_dict(
    torch.load("app/weights/model_weights.pth", map_location=DEVICE)
)
model.eval()


def predict_from_inputs(image, text):
    with torch.no_grad():
        image = image.unsqueeze(0).to(DEVICE)
        logits = model(image, [text])
        probs = torch.softmax(logits, dim=1)

        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()

    return f"Class {pred} (confidence {conf:.2f})"

import torch
from app.fusion_model import MultimodalTriageModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = MultimodalTriageModel(num_classes=3).to(DEVICE)

state = torch.load("weights/model_weights.pth", map_location=DEVICE)
model.load_state_dict(state, strict=True)

model.eval()

@torch.no_grad()
def predict_from_inputs(image, text):
    image = image.unsqueeze(0).to(DEVICE)
    logits = model(image, [text])
    probs = torch.softmax(logits, dim=1)

    pred = probs.argmax(dim=1).item()
    conf = probs.max().item()

    return pred, conf

import os
import torch
from app.fusion_model import MultimodalTriageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "weights/model_weights.pth"

def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing weights at {WEIGHTS_PATH}")

    model = MultimodalTriageModel(num_classes=3)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model()

@torch.no_grad()
def predict_from_inputs(image, text):
    # dummy tokenizer (replace later)
    tokens = torch.randint(0, 10000, (1, 10)).to(DEVICE)
    image = image.unsqueeze(0).to(DEVICE)

    logits = MODEL(image, tokens)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item()

import torch
import os

from src.fusion_model import MultimodalTriageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "model_weights.pth")


def load_model():
    model = MultimodalTriageModel(num_classes=3)

    if os.path.exists(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()


@torch.no_grad()
def predict_from_inputs(image, text):
    tokens = torch.randint(0, 10000, (1, 20)).to(DEVICE)
    image = image.to(DEVICE)

    logits = MODEL(image, tokens)
    probs = torch.softmax(logits, dim=1)

    conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()
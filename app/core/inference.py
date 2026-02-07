import os
import torch
from app.fusion_model import MultimodalTriageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR = "weights"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "model_weights.pth")

# 🔥 PUBLIC HF MODEL URL (replace with your own if needed)
WEIGHTS_URL = (
    "https://huggingface.co/Rhutam/multimodal-medical-triage-model/resolve/main/model_weights.pth"
)

def download_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if not os.path.exists(WEIGHTS_PATH):
        print("⬇️ Downloading model weights...")
        torch.hub.download_url_to_file(
            WEIGHTS_URL,
            WEIGHTS_PATH,
            progress=True
        )
        print("✅ Weights downloaded")

def load_model():
    download_weights()

    model = MultimodalTriageModel(num_classes=3)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model()

@torch.no_grad()
def predict_from_inputs(image, text):
    logits = MODEL(image, text)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()

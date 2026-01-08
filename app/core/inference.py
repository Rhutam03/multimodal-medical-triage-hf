import os
import torch

from ..fusion_model import MultimodalTriageModel
from ..preprocessing.image_preprocess import preprocess_image
from ..preprocessing.text_preprocess import preprocess_text

# --------------------------------------------------
# Device (HF free tier = CPU only)
# --------------------------------------------------
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Resolve absolute path to weights (CRITICAL FIX)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "weights",
    "model_weights.pth"
)

# --------------------------------------------------
# Cache model (lazy load)
# --------------------------------------------------
_model = None


def get_model():
    """
    Load model only once (lazy loading).
    Prevents repeated reloads and HF cold-start crashes.
    """
    global _model

    if _model is not None:
        return _model

    # ---- Safety checks (PREVENT EOFError) ----
    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(f"❌ Weights file not found at: {WEIGHTS_PATH}")

    if os.path.getsize(WEIGHTS_PATH) == 0:
        raise RuntimeError("❌ Weights file exists but is EMPTY (0 bytes)")

    # ---- Initialize model ----
    model = MultimodalTriageModel(num_classes=3)

    try:
        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location=DEVICE
        )
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load weights: {str(e)}")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


def predict(image_path: str, text: str) -> int:
    """
    Run inference on image + text and return triage level.
    """

    # ---- Load model ----
    model = get_model()

    # ---- Image preprocessing ----
    image_tensor = preprocess_image(image_path)

    if image_tensor is None:
        raise RuntimeError("❌ Image preprocessing returned None")

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # ---- Text preprocessing ----
    if text is None or len(text.strip()) == 0:
        text = "No symptoms provided"

    input_ids, attention_mask = preprocess_text(text)

    input_ids = input_ids.unsqueeze(0).to(DEVICE)
    attention_mask = attention_mask.unsqueeze(0).to(DEVICE)

    # ---- Inference ----
    with torch.no_grad():
        outputs = model(
            image_tensor,
            input_ids,
            attention_mask
        )

    # ---- Return class index ----
    return int(outputs.argmax(dim=1).item())

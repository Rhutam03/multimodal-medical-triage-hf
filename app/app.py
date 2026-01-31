import torch
import gradio as gr
from PIL import Image
import sys
import os

# 🔧 Make repo root importable
sys.path.append(os.path.dirname(__file__))

from fusion_model import MultimodalTriageModel
from preprocessing.image_preprocess import preprocess_image
from preprocessing.text_preprocess import preprocess_text

# =====================
# DEVICE
# =====================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

# =====================
# LOAD MODEL
# =====================
MODEL_PATH = "app/weights/triage_best.pt"

model = MultimodalTriageModel(num_classes=3)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

model.to(DEVICE)
model.eval()

# =====================
# LABELS
# =====================
LABELS = ["Low Risk", "Medium Risk", "High Risk"]

# =====================
# PREDICT
# =====================
def predict(image: Image.Image, text: str):
    if image is None:
        return {label: 0.0 for label in LABELS}

    image_tensor = preprocess_image(image).unsqueeze(0).to(DEVICE)
    text_tensor = preprocess_text(text).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor, text_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# =====================
# GRADIO UI
# =====================
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Medical Image"),
        gr.Textbox(
            lines=4,
            placeholder="Enter clinical notes",
            label="Clinical Text"
        ),
    ],
    outputs=gr.Label(label="Triage Prediction"),
    title="Multimodal Medical Triage System",
    description="Image + text based medical triage classification"
)

# =====================
# LAUNCH
# =====================
if __name__ == "__main__":
    interface.launch()

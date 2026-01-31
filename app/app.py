import torch
import gradio as gr
from PIL import Image

from app.fusion_model import MultimodalTriageModel
from app.preprocessing.image_preprocess import preprocess_image
from app.preprocessing.text_preprocess import preprocess_text

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
# PREDICTION FUNCTION
# =====================
def predict(image: Image.Image, text: str):
    """
    image: PIL Image
    text: clinical notes
    """

    if image is None:
        return {label: 0.0 for label in LABELS}

    # ---- Preprocess
    image_tensor = preprocess_image(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
    text_tensor = preprocess_text(text).unsqueeze(0).to(DEVICE)     # [1, 128]

    # ---- Inference
    with torch.no_grad():
        outputs = model(image_tensor, text_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# =====================
# GRADIO INTERFACE
# =====================
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Medical Image"),
        gr.Textbox(
            lines=4,
            placeholder="Enter clinical notes (symptoms, observations, etc.)",
            label="Clinical Text"
        ),
    ],
    outputs=gr.Label(label="Triage Prediction"),
    title="Multimodal Medical Triage System",
    description=(
        "This demo predicts patient triage level using both a medical image "
        "and clinical text notes. The model combines CNN-based image features "
        "with text embeddings for multimodal decision-making."
    ),
    examples=[
        ["examples/sample1.jpg", "Patient has mild headache and dizziness"],
        ["examples/sample2.jpg", "Severe bleeding and loss of consciousness"]
    ]
)

# =====================
# LAUNCH
# =====================
if __name__ == "__main__":
    interface.launch()

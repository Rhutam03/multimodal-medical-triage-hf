import torch
import gradio as gr
from torchvision import transforms
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))
from src.core.inference import predict_from_inputs

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

def predict(image, text):
    image = transform(image).unsqueeze(0)
    pred, conf = predict_from_inputs(image, text)
    return f"Prediction: {pred} | Confidence: {conf:.4f}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy"),
        gr.Textbox(label="Clinical Notes"),
    ],
    outputs="text",
    title="Multimodal Medical Triage"
)

if __name__ == "__main__":
    demo.launch()
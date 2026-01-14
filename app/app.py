import gradio as gr
import torch

from app.core.inference import load_model, predict

model = load_model("app/weights/model_weights.pth")

def run_inference(image, text):
    img_tensor = None
    txt_tensor = None

    if image is not None:
        img_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0

    if text.strip():
        txt_tensor = torch.randn(128)

    return predict(model, img_tensor, txt_tensor)

gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Image(type="numpy", label="Medical Image (optional)"),
        gr.Textbox(label="Medical Description (optional)")
    ],
    outputs=gr.Textbox(label="Triage Result"),
    title="Multimodal Medical Triage System"
).launch()

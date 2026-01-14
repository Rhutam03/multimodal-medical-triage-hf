import gradio as gr
from core.inference import load_model, predict

MODEL_PATH = "app/weights/model_weights.pth"
model = load_model(MODEL_PATH)

def run(image, text):
    return predict(model, image=image, text=text)

gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil", label="Medical Image"),
        gr.Textbox(label="Medical Description"),
    ],
    outputs=gr.Textbox(label="Triage Result"),
    title="Multimodal Medical Triage System",
).launch()

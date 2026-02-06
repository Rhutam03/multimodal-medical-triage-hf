import gradio as gr
from PIL import Image
from app.core.inference import predict_from_inputs
from app.preprocess.image_preprocess import image_transform

def run(image, text):
    image = image_transform(image)
    pred, conf = predict_from_inputs(image, text)
    return f"Class {pred} (confidence {conf:.2f})"

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Clinical description")
    ],
    outputs="text",
    title="Multimodal Medical Triage"
)

if __name__ == "__main__":
    demo.launch()

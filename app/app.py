import gradio as gr
from app.core.inference import predict_from_inputs
from app.preprocess.image_preprocess import image_transform
from PIL import Image


def run(image, text):
    image = image_transform(image)
    probs = predict_from_inputs(image, text)
    return str(probs)


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

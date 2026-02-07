import gradio as gr
from app.core.inference import predict_from_inputs

def run(image, text):
    pred, conf = predict_from_inputs(image, text)
    return f"Prediction: {pred}, Confidence: {conf:.2f}"

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="tensor"),
        gr.Textbox(label="Patient description")
    ],
    outputs="text",
    title="Multimodal Medical Triage"
)

if __name__ == "__main__":
    demo.launch()

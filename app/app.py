import gradio as gr
from core.inference import predict

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Medical Image (optional)"),
        gr.Textbox(label="Medical Description (optional)")
    ],
    outputs=gr.Text(label="Triage Result"),
    title="Multimodal Medical Triage System",
    description="Demo system for multimodal medical triage (image + text)."
)

if __name__ == "__main__":
    demo.launch()

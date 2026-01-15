import gradio as gr
from core.inference import predict_from_inputs


def run(image, text):
    return predict_from_inputs(image=image, text=text)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil", label="Medical Image"),
        gr.Textbox(label="Clinical Description", lines=2),
    ],
    outputs=gr.Textbox(label="Triage Result"),
    title="Multimodal Medical Triage System",
)

if __name__ == "__main__":
    demo.launch()
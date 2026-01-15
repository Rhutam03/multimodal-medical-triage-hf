import gradio as gr

def run(image, text):
    # Import INSIDE function (prevents circular import)
    from app.core.inference import predict_from_inputs
    return predict_from_inputs(image=image, text=text)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil", label="Medical Image"),
        gr.Textbox(label="Clinical Description", placeholder="Optional")
    ],
    outputs=gr.Textbox(label="Triage Result"),
    title="Multimodal Medical Triage System",
    description="ISIC-trained multimodal medical triage model"
)

demo.launch()

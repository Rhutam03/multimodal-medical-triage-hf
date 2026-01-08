import torch
torch.set_num_threads(1)
from fastapi import FastAPI, UploadFile, File, Form
import shutil
from .core.inference import predict
from fastapi import FastAPI
import torch

torch.set_num_threads(1)

app = FastAPI(
    title="Multimodal Medical Triage API",
    root_path="/proxy/7860",
    docs_url="/docs",
    openapi_url="/openapi.json"
)



@app.get("/")
def root():
    return {"status": "ok", "message": "App running"}

@app.post("/predict")
async def predict_api(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    image_path = f"/tmp/{image.filename}"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    triage_level = predict(image_path, text)

    return {"triage_level": int(triage_level)}

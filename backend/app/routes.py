from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.image_utils import InvalidImageError
from app.predictor import run_prediction

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok", "environment": "dev", "model_version": "v1"}


@router.post("/predict")
async def predict(
    image: UploadFile = File(...),
    text: str = Form("")
):
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        return run_prediction(image_bytes, image.content_type, text)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

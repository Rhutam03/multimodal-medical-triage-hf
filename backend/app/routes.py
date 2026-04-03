from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from src.core.inference import predict_from_inputs

router = APIRouter()


@router.post("/api/predict")
@router.post("/predict")
@router.post("/api/analyze")
@router.post("/analyze")
async def predict_route(
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    note_text: str = Form(""),
    notes: str = Form(""),
    age: str = Form(""),
    sex: str = Form(""),
    site: str = Form(""),
):
    """
    Supports both old and new frontend payloads:
    - file OR image
    - note_text OR notes
    - /api/predict OR /predict OR /api/analyze OR /analyze
    """
    try:
        upload = file or image
        if upload is None:
            raise HTTPException(
                status_code=400,
                detail="No image file provided. Expected form field 'file' or 'image'.",
            )

        contents = await upload.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        final_note_text = note_text or notes or ""

        result = predict_from_inputs(
            image=pil_image,
            note_text=final_note_text,
            age=age,
            sex=sex,
            site=site,
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
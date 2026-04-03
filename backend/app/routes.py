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
async def api_predict(
    file: UploadFile = File(...),
    note_text: str = Form(""),
    age: str = Form(""),
    sex: str = Form(""),
    site: str = Form(""),
):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        result = predict_from_inputs(
            image=image,
            note_text=note_text,
            age=age,
            sex=sex,
            site=site,
        )
        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
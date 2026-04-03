from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from src.core.inference import predict_from_inputs

router = APIRouter()

HISTORY_PATH = BACKEND_DIR / "artifacts" / "prediction_history.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
MAX_HISTORY_ITEMS = 25


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(items: list[dict[str, Any]]) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(items[:MAX_HISTORY_ITEMS], f, ensure_ascii=False, indent=2)


@router.get("/api/predictions")
@router.get("/predictions")
def get_predictions():
    return load_history()


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
    Backward-compatible with multiple frontend payload styles:
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

        history = load_history()

        history_item = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_name": upload.filename or "uploaded-image",
            "note_text": final_note_text,
            "triage_level": result.get("triage_level"),
            "confidence": result.get("confidence"),
            "probabilities": result.get("probabilities", {}),
            "predicted_index": result.get("predicted_index"),
        }

        history.insert(0, history_item)
        save_history(history)

        result["request_id"] = history_item["id"]
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
import time
import uuid

from app.image_utils import load_image_tensor
from src.core.inference import LABEL_MAP, predict_from_inputs


def run_prediction(image_bytes: bytes, content_type: str, text: str) -> dict:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    image_tensor = load_image_tensor(image_bytes)

    pred, conf, prob_dict = predict_from_inputs(image_tensor, text or "")
    predicted_class = LABEL_MAP[pred]

    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    warnings: list[str] = []
    if not (text or "").strip():
        warnings.append("Clinical notes were empty; prediction used image input only.")

    return {
        "request_id": request_id,
        "model_version": "v1",
        "predicted_class": predicted_class,
        "class_id": pred,
        "confidence": float(conf),
        "probabilities": {k: float(v) for k, v in prob_dict.items()},
        "warnings": warnings,
        "image_url": None,
        "duration_ms": duration_ms,
        "saved": False,
    }

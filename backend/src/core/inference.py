from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[2]   
REPO_ROOT = CURRENT_FILE.parents[3]     
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.fusion_model import MultimodalTriageModel
from src.preprocess.image_preprocess import image_transform
from src.preprocess.text_preprocess import (
    build_vocab_from_csv,
    encode_text,
    load_vocab,
    preprocess_text,
    save_vocab,
)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

WEIGHTS_CANDIDATES = [
    BACKEND_DIR / "artifacts" / "model_weights.pth",
    BACKEND_DIR / "weights" / "model_weights.pth",
    REPO_ROOT / "artifacts" / "model_weights.pth",
    REPO_ROOT / "weights" / "model_weights.pth",
]

VOCAB_CANDIDATES = [
    BACKEND_DIR / "artifacts" / "vocab.json",
    BACKEND_DIR / "weights" / "vocab.json",
    REPO_ROOT / "artifacts" / "vocab.json",
    REPO_ROOT / "weights" / "vocab.json",
]

LABELS_CSV_CANDIDATES = [
    BACKEND_DIR / "data" / "labels.csv",
    REPO_ROOT / "data" / "labels.csv",
]

LABEL_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

DEFAULT_CANONICAL_TEXT = (
    "age unknown. sex unknown. site unknown. "
    "symptoms unknown. change unknown. history unknown."
)

_MODEL: MultimodalTriageModel | None = None
_VOCAB: dict[str, int] | None = None
_MAX_LEN: int = 48
_WEIGHTS_PATH: Path | None = None
_VOCAB_PATH: Path | None = None
_LABELS_CSV_PATH: Path | None = None


def _find_existing_file(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _is_valid_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    if path.stat().st_size == 0:
        return False

    head = path.read_bytes()[:200]
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False

    return True


def _find_valid_checkpoint(candidates: list[Path]) -> Path:
    for path in candidates:
        if _is_valid_checkpoint(path):
            return path

    details = []
    for path in candidates:
        exists = path.exists()
        size = path.stat().st_size if exists and path.is_file() else "missing"
        details.append(f"{path} -> exists={exists}, size={size}")

    raise FileNotFoundError(
        "Could not find a valid model checkpoint. Checked:\n" + "\n".join(details)
    )


def _extract_state_dict_and_max_len(
    checkpoint_obj: Any,
) -> tuple[dict[str, torch.Tensor], int]:
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state_dict"]
        max_len = int(checkpoint_obj.get("max_len", 48))
        return state_dict, max_len

    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj, 48

    raise ValueError("Unsupported checkpoint format.")


def _load_or_rebuild_vocab(
    vocab_candidates: list[Path],
    labels_csv_candidates: list[Path],
) -> tuple[dict[str, int], Path, Path]:
    for vocab_path in vocab_candidates:
        if vocab_path.exists():
            try:
                vocab = load_vocab(vocab_path)
                if isinstance(vocab, dict) and len(vocab) > 0:
                    labels_csv_path = _find_existing_file(labels_csv_candidates)
                    if labels_csv_path is None:
                        raise FileNotFoundError("labels.csv not found while loading vocab.")
                    print(f"Loaded vocab from: {vocab_path}")
                    return vocab, vocab_path, labels_csv_path
            except Exception as exc:
                print(f"Failed to load vocab from {vocab_path}: {exc}")

    labels_csv_path = _find_existing_file(labels_csv_candidates)
    if labels_csv_path is None:
        raise FileNotFoundError(
            "Could not find labels.csv in any expected location:\n"
            + "\n".join(str(p) for p in labels_csv_candidates)
        )

    vocab = build_vocab_from_csv(labels_csv_path, text_column="text")
    target_vocab_path = vocab_candidates[0]
    target_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, target_vocab_path)

    print(f"Rebuilt and saved vocab to: {target_vocab_path}")
    return vocab, target_vocab_path, labels_csv_path


def _clean_field(value: str | int | None, unknown: str = "unknown") -> str:
    if value is None:
        return unknown
    value = str(value).strip()
    if not value:
        return unknown
    cleaned = preprocess_text(value)
    return cleaned if cleaned else unknown


def canonicalize_user_text(
    note_text: str | None = "",
    age: str | int | None = None,
    sex: str | None = None,
    site: str | None = None,
) -> str:
    age_value = _clean_field(age, "unknown")
    sex_value = _clean_field(sex, "unknown")
    site_value = _clean_field(site, "unknown")
    note_value = _clean_field(note_text, "unknown")

    return (
        f"age {age_value}. "
        f"sex {sex_value}. "
        f"site {site_value}. "
        f"symptoms unknown. "
        f"change unknown. "
        f"history unknown. "
        f"user note {note_value}."
    )


def _get_runtime() -> tuple[MultimodalTriageModel, dict[str, int], int]:
    global _MODEL, _VOCAB, _MAX_LEN, _WEIGHTS_PATH, _VOCAB_PATH, _LABELS_CSV_PATH

    if _MODEL is not None and _VOCAB is not None:
        return _MODEL, _VOCAB, _MAX_LEN

    vocab, vocab_path, labels_csv_path = _load_or_rebuild_vocab(
        VOCAB_CANDIDATES,
        LABELS_CSV_CANDIDATES,
    )

    weights_path = _find_valid_checkpoint(WEIGHTS_CANDIDATES)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict, max_len = _extract_state_dict_and_max_len(checkpoint)

    model = MultimodalTriageModel(
        vocab_size=len(vocab),
        num_classes=3,
    ).to(DEVICE)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _MODEL = model
    _VOCAB = vocab
    _MAX_LEN = max_len
    _WEIGHTS_PATH = weights_path
    _VOCAB_PATH = vocab_path
    _LABELS_CSV_PATH = labels_csv_path

    print(f"Loaded checkpoint from: {_WEIGHTS_PATH}")
    print(f"Loaded vocab from: {_VOCAB_PATH}")
    print(f"Using labels CSV: {_LABELS_CSV_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Max token length: {_MAX_LEN}")

    return _MODEL, _VOCAB, _MAX_LEN


def predict_from_inputs(
    image: Image.Image | str | Path,
    note_text: str | None = "",
    age: str | int | None = None,
    sex: str | None = None,
    site: str | None = None,
) -> dict[str, Any]:
    model, vocab, max_len = _get_runtime()

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image, file path string, or Path.")

    image = image.convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    canonical_text = canonicalize_user_text(
        note_text=note_text,
        age=age,
        sex=sex,
        site=site,
    )

    token_ids = torch.tensor(
        encode_text(canonical_text, vocab, max_len),
        dtype=torch.long,
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor, token_ids)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    return {
        "predicted_index": pred_idx,
        "triage_level": LABEL_MAP[pred_idx],
        "confidence": confidence,
        "probabilities": {
            LABEL_MAP[0]: float(probs[0]),
            LABEL_MAP[1]: float(probs[1]),
            LABEL_MAP[2]: float(probs[2]),
        },
        "text_used": canonical_text,
        "model_info": {
            "device": str(DEVICE),
            "weights_path": str(_WEIGHTS_PATH) if _WEIGHTS_PATH else None,
            "vocab_path": str(_VOCAB_PATH) if _VOCAB_PATH else None,
            "labels_csv_path": str(_LABELS_CSV_PATH) if _LABELS_CSV_PATH else None,
            "max_len": max_len,
        },
    }


def warmup_runtime() -> None:
    _get_runtime()
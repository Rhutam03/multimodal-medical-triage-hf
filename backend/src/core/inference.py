from pathlib import Path
import torch

from src.fusion_model import MultimodalTriageModel
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

CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[3]

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


def _load_or_rebuild_vocab(
    vocab_candidates: list[Path],
    labels_candidates: list[Path],
) -> tuple[dict[str, int], Path]:
    last_error: Exception | None = None

    for vocab_path in vocab_candidates:
        if vocab_path.exists():
            try:
                vocab = load_vocab(str(vocab_path))
                print(f"Loaded vocab from: {vocab_path}")
                return vocab, vocab_path
            except Exception as exc:
                last_error = exc
                print(f"Failed to load vocab from {vocab_path}: {exc}")

    labels_path = None
    for candidate in labels_candidates:
        if candidate.exists():
            labels_path = candidate
            break

    if labels_path is None:
        checked = "\n".join(str(p) for p in labels_candidates)
        raise FileNotFoundError(
            f"Could not rebuild vocab because no labels.csv was found. Checked:\n{checked}"
        ) from last_error

    save_path = BACKEND_DIR / "artifacts" / "vocab.json"
    vocab = build_vocab_from_csv(str(labels_path))
    save_vocab(vocab, str(save_path))

    print(f"Rebuilt vocab from: {labels_path}")
    print(f"Saved rebuilt vocab to: {save_path}")

    return vocab, save_path


WEIGHTS_PATH = _find_valid_checkpoint(WEIGHTS_CANDIDATES)
VOCAB, VOCAB_PATH = _load_or_rebuild_vocab(VOCAB_CANDIDATES, LABELS_CSV_CANDIDATES)


def load_model_and_vocab():
    model = MultimodalTriageModel(
        vocab_size=len(VOCAB),
        num_classes=3,
    )

    try:
        state = torch.load(str(WEIGHTS_PATH), map_location=DEVICE)
    except EOFError as exc:
        raise ValueError(
            f"Model checkpoint at {WEIGHTS_PATH} is incomplete or corrupted."
        ) from exc

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"Using device: {DEVICE}")
    print(f"Loaded weights from: {WEIGHTS_PATH}")
    print(f"Using vocab from: {VOCAB_PATH}")

    return model, VOCAB


MODEL, VOCAB = load_model_and_vocab()


@torch.no_grad()
def predict_from_inputs(image, text):
    text = preprocess_text(text or "")

    token_ids = torch.tensor(
        [encode_text(text, VOCAB, max_len=20)],
        dtype=torch.long,
        device=DEVICE,
    )

    image = image.to(DEVICE)

    logits = MODEL(image, token_ids)
    probs = torch.softmax(logits, dim=1)[0]

    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())

    prob_dict = {
        LABEL_MAP[i]: float(probs[i].item())
        for i in range(len(LABEL_MAP))
    }

    return pred, conf, prob_dict
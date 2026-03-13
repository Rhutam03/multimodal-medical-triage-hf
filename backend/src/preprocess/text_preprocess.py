import csv
import json
import re
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_LEN = 20


def preprocess_text(text: str) -> str:
    if not text or not text.strip():
        return "no clinical description provided"
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    text = preprocess_text(text)
    return re.findall(r"\b[a-z0-9]+\b", text)


def build_vocab(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab: dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = MAX_LEN) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]

    if len(ids) < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


def save_vocab(vocab: dict[str, int], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: str) -> dict[str, int]:
    vocab_path = Path(path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    if vocab_path.stat().st_size == 0:
        raise ValueError(f"Vocab file is empty: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Vocab file must contain a JSON object: {vocab_path}")

    if PAD_TOKEN not in data or UNK_TOKEN not in data:
        raise ValueError(
            f"Vocab file is missing required tokens {PAD_TOKEN} and/or {UNK_TOKEN}: {vocab_path}"
        )

    return {str(k): int(v) for k, v in data.items()}


def build_vocab_from_csv(labels_csv_path: str, text_column: str | None = None) -> dict[str, int]:
    csv_path = Path(labels_csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    texts: list[str] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {csv_path}")

        candidate_columns = ["text", "notes", "clinical_notes", "description"]
        chosen_column = text_column

        if chosen_column is None:
            for col in candidate_columns:
                if col in reader.fieldnames:
                    chosen_column = col
                    break

        if chosen_column is None:
            raise ValueError(
                f"Could not find a usable text column in {csv_path}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            texts.append((row.get(chosen_column) or "").strip())

    return build_vocab(texts)
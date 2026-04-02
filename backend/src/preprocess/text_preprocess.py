from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def preprocess_text(text: str | None) -> str:
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def build_vocab_from_csv(
    csv_path: str | Path,
    text_column: str = "text",
    min_freq: int = 1,
) -> dict[str, int]:
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").astype(str).tolist()
    return build_vocab(texts, min_freq=min_freq)


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]

    if len(ids) < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


def save_vocab(vocab: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"Invalid vocab file: {path}")

    return vocab
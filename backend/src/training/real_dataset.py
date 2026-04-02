from __future__ import annotations

import os
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[2]  # .../backend
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocess.text_preprocess import encode_text, preprocess_text

DEFAULT_TEXT = (
    "age unknown. sex unknown. site unknown. "
    "symptoms unknown. change unknown. history unknown."
)


class RealMultimodalDataset(Dataset):
    def __init__(self, labels_csv, image_dir, transform, vocab, max_len=48):
        self.df = pd.read_csv(labels_csv).copy()
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        self.max_len = max_len

        required_cols = {"image", "text", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"labels.csv is missing required columns: {missing}")

        self.df["image"] = self.df["image"].astype(str)
        self.df["text"] = self.df["text"].fillna("").astype(str)
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row["image"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image file: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        raw_text = row["text"].strip()
        text = preprocess_text(raw_text) if raw_text else DEFAULT_TEXT
        if not text:
            text = DEFAULT_TEXT

        token_ids = torch.tensor(
            encode_text(text, self.vocab, self.max_len),
            dtype=torch.long,
        )

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return image, token_ids, label
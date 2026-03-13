import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocess.text_preprocess import encode_text


class RealMultimodalDataset(Dataset):
    def __init__(self, labels_csv, image_dir, transform, vocab, max_len=20):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = row["text"]
        token_ids = torch.tensor(
            encode_text(text, self.vocab, self.max_len),
            dtype=torch.long
        )

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return image, token_ids, label
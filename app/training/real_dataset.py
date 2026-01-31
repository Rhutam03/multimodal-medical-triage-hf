import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from app.preprocessing.image_preprocess import preprocess_image
from app.preprocessing.text_preprocess import preprocess_text


class RealMultimodalDataset(Dataset):
    def __init__(self, labels_csv, image_dir):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = image_dir

        assert len(self.df) > 0, "labels.csv is empty"
        print("✅ Dataset loaded")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row["image"])
        label = int(row["label"])
        text = row.get("report", "")

        # ---- Image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess_image(image)   # [3, 224, 224]

        # ---- Text
        text_tensor = preprocess_text(text)       # [128]

        return (
            image_tensor,
            text_tensor,
            torch.tensor(label, dtype=torch.long)
        )

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RealMultimodalDataset(Dataset):
    def __init__(self, labels_csv: str, image_dir: str):
        """
        labels_csv: path to data/labels.csv
        image_dir: path to data/images/
        """
        self.df = pd.read_csv(labels_csv)
        self.image_dir = image_dir

        if len(self.df) == 0:
            raise RuntimeError("❌ labels.csv is empty")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ----- IMAGE -----
        img_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # ----- TEXT -----
        text = row["text"]          # str (NOT tuple)

        # ----- LABEL -----
        label = int(row["label"])

        return {
            "image": image,
            "text": text,
            "label": label
        }

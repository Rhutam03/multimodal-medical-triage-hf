import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision import transforms


class RealMultimodalDataset(Dataset):
    def __init__(self, labels_csv, image_dir):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = Path(image_dir)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(self.image_dir / row["image"]).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "text": row["text"],
            "label": int(row["label"])
        }

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os

class RealMultimodalDataset(Dataset):
    def __init__(self, labels_csv, image_dir, transform):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row["image"])).convert("RGB")
        image = self.transform(image)

        text = row["text"]
        label = int(row["label"])

        return image, text, label

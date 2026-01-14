import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RealMultimodalDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Text → fake embedding (simple + stable)
        text_embedding = torch.randn(128)

        label = int(row["label"])

        return image, text_embedding, label

import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from app.preprocess.image_preprocess import image_transform
from app.preprocess.text_preprocess import preprocess_text


class RealMultimodalDataset(Dataset):
    def __init__(self, labels_csv: str, image_dir: str):
        self.df = pd.read_csv(labels_csv)
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(self.image_dir / row["image"]).convert("RGB")
        image = image_transform(image)

        text = preprocess_text(row.get("text", ""))
        label = int(row["label"])

        return image, text, label

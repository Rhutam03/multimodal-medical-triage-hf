import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fusion_model import MultimodalTriageModel
from training.real_dataset import RealMultimodalDataset
from preprocess.image_preprocess import image_transform

DEVICE = torch.device("cpu")


def train():
    dataset = RealMultimodalDataset(
        labels_csv="data/labels.csv",
        image_dir="data/images",
        transform=image_transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultimodalTriageModel(num_classes=3).to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(3):
        total_loss = 0
        for images, texts, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(images, texts), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss {total_loss:.4f}")

    torch.save(model.state_dict(), "weights/model_weights.pth")
    print("Training complete. Weights saved.")


if __name__ == "__main__":
    train()

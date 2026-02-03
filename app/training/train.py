import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from app.fusion_model import MultimodalTriageModel
from app.training.real_dataset import RealMultimodalDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = MultimodalTriageModel(num_classes=3).to(DEVICE)
    model.train()

    dataset = RealMultimodalDataset(
        labels_csv="data/labels.csv",
        image_dir="data/images"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    num_epochs = 5

    for epoch in range(num_epochs):
        total_loss = 0.0

        for images, texts, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images, texts)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "app/weights/model_weights.pth")
    print("✅ Training complete. Weights saved.")


if __name__ == "__main__":
    train()

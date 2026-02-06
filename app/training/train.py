import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from app.fusion_model import MultimodalTriageModel
from app.training.real_dataset import RealMultimodalDataset
from app.preprocess.image_preprocess import image_transform

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train():
    model = MultimodalTriageModel(num_classes=3).to(DEVICE)
    model.train()

    dataset = RealMultimodalDataset(
        labels_csv="data/labels.csv",
        image_dir="data/images",
        transform=image_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(3):
        total_loss = 0
        for images, texts, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images, texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/3 | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "weights/model_weights.pth")
    print("✅ Training complete. Weights saved.")

if __name__ == "__main__":
    train()

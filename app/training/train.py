import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.fusion_model import MultimodalTriageModel
from app.training.real_dataset import RealMultimodalDataset

DEVICE = torch.device("cpu")

def train():
    print("🚀 Starting training...")

    dataset = RealMultimodalDataset(
        csv_path="data/labels.csv",
        image_dir="data/images"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    model = MultimodalTriageModel(num_classes=3)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for epoch in range(3):
        total_loss = 0.0

        for images, texts, labels in dataloader:
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # ✅ CORRECT CALL
            outputs = model(image=images, text=texts)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # ---- SAVE WEIGHTS ----
    weights_dir = "app/weights"
    os.makedirs(weights_dir, exist_ok=True)

    weights_path = os.path.join(weights_dir, "model_weights.pth")
    torch.save(model.state_dict(), weights_path)

    print("✅ model_weights.pth saved:", weights_path)
    print("📦 Size:", os.path.getsize(weights_path), "bytes")


if __name__ == "__main__":
    train()

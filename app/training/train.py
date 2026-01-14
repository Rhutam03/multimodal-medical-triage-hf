import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from app.fusion_model import MultimodalTriageModel

DEVICE = torch.device("cpu")


class DummyMultimodalDataset(Dataset):
    def __len__(self):
        return 50

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        text = torch.randn(128)
        label = torch.randint(0, 3, (1,)).item()
        return image, text, label


def train():
    print("🚀 Starting training...")

    dataset = DummyMultimodalDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MultimodalTriageModel(num_classes=3).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for epoch in range(3):
        total_loss = 0.0

        for image, text, label in dataloader:
            image = image.to(DEVICE)
            text = text.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(image=image, text=text)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    weights_path = os.path.join(weights_dir, "model_weights.pth")
    torch.save(model.state_dict(), weights_path)

    print("✅ model_weights.pth saved at:", weights_path)
    print("📦 File size:", os.path.getsize(weights_path), "bytes")


if __name__ == "__main__":
    train()

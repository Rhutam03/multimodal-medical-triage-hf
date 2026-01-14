import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# IMPORTANT: run with `python -m app.training.train`
from app.fusion_model import MultimodalTriageModel

# --------------------------------------------------
# Device
# --------------------------------------------------
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Dummy Dataset (matches model.forward(image, text))
# --------------------------------------------------
class DummyMultimodalDataset(Dataset):
    def __len__(self):
        return 50

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)   # fake image tensor
        text = torch.randn(128)            # fake text embedding
        label = torch.randint(0, 3, (1,)).item()
        return image, text, label


# --------------------------------------------------
# Training Function
# --------------------------------------------------
def train():
    print("🚀 Starting training...")

    dataset = DummyMultimodalDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MultimodalTriageModel(num_classes=3)
    model.to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):
        total_loss = 0.0

        for image, text, label in dataloader:
            image = image.to(DEVICE)
            text = text.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            # ✅ CORRECT CALL (THIS WAS THE BUG)
            outputs = model(image=image, text=text)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # --------------------------------------------------
    # Save weights
    # --------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "model_weights.pth")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)

    print("✅ model_weights.pth saved at:", WEIGHTS_PATH)
    print("📦 File size:", os.path.getsize(WEIGHTS_PATH), "bytes")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    train()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# IMPORTANT: relative import (because we run with -m)
from ..fusion_model import MultimodalTriageModel

# --------------------------------------------------
# Device (CPU for HF + local safety)
# --------------------------------------------------
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Dummy Dataset (VALID + SAFE)
# --------------------------------------------------
class DummyMultimodalDataset(Dataset):
    """
    Generates FAKE but VALID multimodal data
    so training really runs and produces real weights.
    """

    def __len__(self):
        return 50

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)          # fake image
        input_ids = torch.randint(0, 1000, (128,))  # fake tokens
        attention_mask = torch.ones(128)
        label = torch.randint(0, 3, (1,)).item()   # 3 classes
        return image, input_ids, attention_mask, label


# --------------------------------------------------
# Training Function
# --------------------------------------------------
def train():
    print("🚀 Starting training...")

    dataset = DummyMultimodalDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MultimodalTriageModel(num_classes=3)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for epoch in range(3):
        total_loss = 0.0

        for image, input_ids, attention_mask, label in dataloader:
            image = image.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(image, input_ids, attention_mask)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # --------------------------------------------------
    # SAVE WEIGHTS (CRITICAL FIX)
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

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# IMPORTANT: absolute import (because we run with -m)
from app.fusion_model import MultimodalTriageModel

# --------------------------------------------------
# Device (CPU for Hugging Face + local safety)
# --------------------------------------------------
DEVICE = torch.device("cpu")

# --------------------------------------------------
# Dummy Multimodal Dataset (VALID + SAFE)
# --------------------------------------------------
class DummyMultimodalDataset(Dataset):
    """
    Generates FAKE but VALID multimodal data
    so training actually runs and produces real weights.
    """

    def __len__(self):
        return 50

    def __getitem__(self, idx):
        # Fake image tensor (matches ImageEncoder expectations)
        image = torch.randn(3, 224, 224)

        # Fake text embedding (matches TextEncoder output size)
        # NOTE: This avoids tokenizer/BERT complexity during training
        text = torch.randn(256)

        # 3-class triage label: 0=low, 1=medium, 2=high
        label = torch.randint(0, 3, (1,)).item()

        return image, text, label


# --------------------------------------------------
# Training Function
# --------------------------------------------------
def train():
    print("🚀 Starting training...")

    # Dataset & DataLoader
    dataset = DummyMultimodalDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    model = MultimodalTriageModel(num_classes=3)
    model.to(DEVICE)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    for epoch in range(3):
        total_loss = 0.0

        for image, text, label in dataloader:
            image = image.to(DEVICE)
            text = text.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            # IMPORTANT: forward signature matches fusion_model.py
            outputs = model(image=image, text=text)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/3 - Loss: {total_loss:.4f}")

    # --------------------------------------------------
    # SAVE WEIGHTS (CRITICAL)
    # --------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "model_weights.pth")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    torch.save(model.state_dict(), WEIGHTS_PATH)

    print("✅ Training complete.")
    print("💾 model_weights.pth saved at:", WEIGHTS_PATH)
    print("📦 File size:", os.path.getsize(WEIGHTS_PATH), "bytes")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    train()

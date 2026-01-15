import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.training.real_dataset import RealMultimodalDataset
from app.fusion_model import MultimodalTriageModel

# =============================
# CONFIG
# =============================
BATCH_SIZE = 32            # Increase if memory allows (64 is OK on M4 Pro)
EPOCHS = 3
LR = 1e-4
NUM_WORKERS = 4            # Apple Silicon sweet spot
LABELS_CSV = "data/labels.csv"
IMAGE_DIR = "data/images"
WEIGHTS_OUT = "app/weights/model_weights.pth"

# =============================
# DEVICE (Apple Silicon)
# =============================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# =============================
# COLLATE FUNCTION (CRITICAL FIX)
# =============================
def collate_fn(batch):
    """
    Fixes:
    - text being returned as tuples
    - ensures images are tensors
    """
    images = []
    texts = []
    labels = []

    for item in batch:
        images.append(item["image"])
        texts.append(item["text"])     # keep as list[str]
        labels.append(item["label"])

    images = torch.stack(images)       # [B, 3, H, W]
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "image": images,
        "text": texts,
        "label": labels
    }

# =============================
# TRAIN LOOP
# =============================
def train():
    # Dataset
    dataset = RealMultimodalDataset(
        labels_csv=LABELS_CSV,
        image_dir=IMAGE_DIR
    )

    if len(dataset) == 0:
        raise RuntimeError("❌ Dataset is empty. Check labels.csv and images folder.")

    print(f"📦 Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=False,          # REQUIRED for MPS
        collate_fn=collate_fn
    )

    # Model
    model = MultimodalTriageModel(num_classes=3)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # =============================
    # TRAINING
    # =============================
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            texts = batch["text"]           # keep as list[str]
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(image=images, text=texts)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    # =============================
    # SAVE MODEL
    # =============================
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"💾 Model saved to {WEIGHTS_OUT}")


# =============================
# ENTRYPOINT (HF SAFE)
# =============================
if __name__ == "__main__":
    train()

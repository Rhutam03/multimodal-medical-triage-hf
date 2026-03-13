import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Subset

from src.fusion_model import MultimodalTriageModel
from src.preprocess.image_preprocess import image_transform
from src.preprocess.text_preprocess import build_vocab, save_vocab
from src.training.real_dataset import RealMultimodalDataset

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REPO_ROOT = os.path.dirname(BASE_DIR)

DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(REPO_ROOT, "data"),
]

DATA_DIR = None
for candidate in DATA_CANDIDATES:
    if os.path.exists(candidate):
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    raise FileNotFoundError("Could not find data directory.")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "model_weights.pth")
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")

NUM_EPOCHS = 3
BATCH_SIZE = 32
SUBSET_SIZE = 2000


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, tokens, labels in loader:
            images = images.to(DEVICE)
            tokens = tokens.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, tokens)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(LABELS_CSV)

    text_column = "text" if "text" in df.columns else df.columns[1]
    vocab = build_vocab(df[text_column].fillna("").astype(str).tolist())
    save_vocab(vocab, VOCAB_PATH)

    dataset = RealMultimodalDataset(
        labels_csv=LABELS_CSV,
        image_dir=IMAGE_DIR,
        transform=image_transform,
        vocab=vocab,
        max_len=20,
    )

    if SUBSET_SIZE > 0:
        subset_size = min(SUBSET_SIZE, len(dataset))
        dataset = Subset(dataset, range(subset_size))
        print(f"Using fast subset: {subset_size} samples")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultimodalTriageModel(vocab_size=len(vocab), num_classes=3).to(DEVICE)

    for param in model.image_encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()

        model.train()
        total_loss = 0.0

        for images, tokens, labels in train_loader:
            images = images.to(DEVICE)
            tokens = tokens.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / max(len(train_loader.dataset), 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        epoch_seconds = time.perf_counter() - epoch_start
        remaining_epochs = NUM_EPOCHS - (epoch + 1)
        eta_seconds = remaining_epochs * epoch_seconds

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Epoch Time: {epoch_seconds:.1f}s | "
            f"ETA: {eta_seconds:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"Saved best model to: {WEIGHTS_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    train()
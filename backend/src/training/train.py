from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[2]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

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

SEED = 42
BATCH_SIZE = int(os.getenv("TRIAGE_BATCH_SIZE", "16"))
NUM_EPOCHS = int(os.getenv("TRIAGE_NUM_EPOCHS", "12"))
VAL_FRACTION = 0.2
MAX_LEN = 48
WEIGHT_DECAY = 1e-4
LOG_EVERY = int(os.getenv("TRIAGE_LOG_EVERY", "25"))
SUBSET_SIZE = int(os.getenv("TRIAGE_SUBSET", "0"))
NUM_WORKERS = int(os.getenv("TRIAGE_NUM_WORKERS", "0"))

DATA_CANDIDATES = [
    BACKEND_DIR / "data",
    REPO_ROOT / "data",
]

DATA_DIR = None
for candidate in DATA_CANDIDATES:
    if candidate.exists():
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    raise FileNotFoundError("Could not find data directory.")

ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS_CSV = DATA_DIR / "labels.csv"
IMAGE_DIR = DATA_DIR / "images"
WEIGHTS_PATH = ARTIFACTS_DIR / "model_weights.pth"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
METRICS_PATH = ARTIFACTS_DIR / "best_metrics.json"
SUBSET_LABELS_PATH = ARTIFACTS_DIR / "labels_subset.csv"

LABEL_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels):
    counts = torch.bincount(torch.tensor(labels), minlength=3).float()
    weights = counts.sum() / (len(counts) * counts.clamp_min(1.0))
    return weights.to(DEVICE), counts.tolist()


def maybe_make_subset(df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    if SUBSET_SIZE <= 0 or SUBSET_SIZE >= len(df):
        return df.reset_index(drop=True), LABELS_CSV

    sampled_df, _ = train_test_split(
        df,
        train_size=SUBSET_SIZE,
        stratify=df["label"],
        random_state=SEED,
        shuffle=True,
    )
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df.to_csv(SUBSET_LABELS_PATH, index=False)

    print(f"\nUsing debug subset: {len(sampled_df)} rows")
    print(sampled_df["label"].value_counts().sort_index())
    print(sampled_df["label"].value_counts(normalize=True).sort_index())
    sys.stdout.flush()

    return sampled_df, SUBSET_LABELS_PATH


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    print("Running validation...")
    sys.stdout.flush()

    with torch.no_grad():
        for batch_idx, (images, tokens, labels) in enumerate(loader, start=1):
            images = images.to(DEVICE)
            tokens = tokens.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, tokens)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            if batch_idx == 1:
                print(
                    f"Validation first batch ok | "
                    f"images={tuple(images.shape)} tokens={tuple(tokens.shape)} labels={tuple(labels.shape)}"
                )
                sys.stdout.flush()

    total = max(len(all_labels), 1)
    avg_loss = total_loss / total
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=[LABEL_MAP[0], LABEL_MAP[1], LABEL_MAP[2]],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "report": report,
        "confusion_matrix": cm,
    }


def train():
    set_seed()

    print(f"Using device: {DEVICE}")
    print(f"Backend dir: {BACKEND_DIR}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Labels CSV: {LABELS_CSV}")
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Artifacts dir: {ARTIFACTS_DIR}")
    print(
        f"Config | batch_size={BATCH_SIZE} epochs={NUM_EPOCHS} "
        f"max_len={MAX_LEN} subset={SUBSET_SIZE} log_every={LOG_EVERY}"
    )
    sys.stdout.flush()

    df = pd.read_csv(LABELS_CSV).copy()

    required_cols = {"image", "text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing required columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nClass proportions:")
    print(df["label"].value_counts(normalize=True).sort_index())
    sys.stdout.flush()

    working_df, working_labels_csv = maybe_make_subset(df)

    vocab = build_vocab(working_df["text"].tolist())
    save_vocab(vocab, VOCAB_PATH)
    print(f"\nSaved vocab to: {VOCAB_PATH}")
    print(f"Vocab size: {len(vocab)}")
    sys.stdout.flush()

    dataset = RealMultimodalDataset(
        labels_csv=str(working_labels_csv),
        image_dir=str(IMAGE_DIR),
        transform=image_transform,
        vocab=vocab,
        max_len=MAX_LEN,
    )

    print(f"Dataset size: {len(dataset)}")
    sys.stdout.flush()

    indices = np.arange(len(working_df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_FRACTION,
        stratify=working_df["label"],
        random_state=SEED,
        shuffle=True,
    )

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    train_labels = working_df.iloc[train_idx]["label"].tolist()
    class_weights, class_counts = compute_class_weights(train_labels)

    print("\nTraining class counts:", class_counts)
    print("Class weights:", class_weights.detach().cpu().tolist())
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    sys.stdout.flush()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("\nBuilding model...")
    sys.stdout.flush()

    model = MultimodalTriageModel(
        vocab_size=len(vocab),
        num_classes=3,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": 1e-4},
            {"params": model.text_encoder.parameters(), "lr": 3e-4},
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not n.startswith("image_encoder.")
                    and not n.startswith("text_encoder.")
                ],
                "lr": 3e-4,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_macro_f1 = -1.0

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()
        model.train()
        total_train_loss = 0.0

        print(f"\nStarting epoch {epoch + 1}/{NUM_EPOCHS} ...")
        sys.stdout.flush()

        for batch_idx, (images, tokens, labels) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            tokens = tokens.to(DEVICE)
            labels = labels.to(DEVICE)

            if batch_idx == 1:
                print(
                    f"First training batch ok | "
                    f"images={tuple(images.shape)} tokens={tuple(tokens.shape)} labels={tuple(labels.shape)}"
                )
                sys.stdout.flush()

            optimizer.zero_grad()
            outputs = model(images, tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)

            if batch_idx % LOG_EVERY == 0 or batch_idx == len(train_loader):
                running_loss = total_train_loss / (batch_idx * BATCH_SIZE)
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Running Train Loss: {running_loss:.4f}"
                )
                sys.stdout.flush()

        train_loss = total_train_loss / max(len(train_loader.dataset), 1)

        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics["macro_f1"])

        epoch_seconds = time.perf_counter() - epoch_start

        low_recall = val_metrics["report"]["Low Risk"]["recall"]
        med_recall = val_metrics["report"]["Medium Risk"]["recall"]
        high_recall = val_metrics["report"]["High Risk"]["recall"]

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} COMPLETE | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Balanced Acc: {val_metrics['balanced_acc']:.4f} | "
            f"Recall L/M/H: {low_recall:.4f}/{med_recall:.4f}/{high_recall:.4f} | "
            f"Time: {epoch_seconds:.1f}s"
        )
        sys.stdout.flush()

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]

            best_payload = {
                "model_state_dict": model.state_dict(),
                "label_map": LABEL_MAP,
                "max_len": MAX_LEN,
                "vocab_size": len(vocab),
                "best_val_macro_f1": best_macro_f1,
                "class_weights": class_weights.detach().cpu().tolist(),
                "train_class_counts": class_counts,
            }

            torch.save(best_payload, WEIGHTS_PATH)

            with open(METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)

            print(f"Saved best model to: {WEIGHTS_PATH}")
            print(f"Saved best metrics to: {METRICS_PATH}")
            sys.stdout.flush()

        if DEVICE.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    print("\nTraining complete.")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    train()
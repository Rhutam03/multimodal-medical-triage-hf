import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import json

from fusion_model import MultimodalTriageModel
from training.real_dataset import RealMultimodalDataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text": [b["text"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch])
    }

def train():
    dataset = RealMultimodalDataset(
        "data/labels.csv",
        "data/images"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = MultimodalTriageModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        preds, labels = [], []

        for batch in loader:
            optimizer.zero_grad()

            out = model(
                image=batch["image"].to(DEVICE),
                text=batch["text"]
            )

            loss = criterion(out, batch["label"].to(DEVICE))
            loss.backward()
            optimizer.step()

            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(batch["label"].numpy())

        acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch+1} | Acc {acc:.3f}")

    torch.save(model.state_dict(), "app/weights/model_weights.pth")

    np.save("app/weights/confusion_matrix.npy",
            confusion_matrix(labels, preds))

    with open("app/weights/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

if __name__ == "__main__":
    train()

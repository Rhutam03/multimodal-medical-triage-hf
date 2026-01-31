import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from app.fusion_model import MultimodalTriageModel
from app.training.real_dataset import RealMultimodalDataset

# =====================
# DEVICE
# =====================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

# =====================
# TRAIN FUNCTION
# =====================
def train():
    dataset = RealMultimodalDataset(
        labels_csv="data/labels.csv",
        image_dir="data/images"
    )

    # ---- Train / Val split
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = MultimodalTriageModel(num_classes=3).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    EPOCHS = 3

    for epoch in range(EPOCHS):
        print(f"\n🟢 Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0

        for step, (images, text_feats, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            text_feats = text_feats.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, text_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 200 == 0:
                print(f"  Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"✅ Train Loss: {avg_train_loss:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        # ---- Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_accuracy": val_acc
            }, "app/weights/triage_best.pt")
            print("💾 Saved best model")

    print("🎉 Training completed")


# =====================
# VALIDATION
# =====================
def validate(model, dataloader, criterion):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, text_feats, labels in dataloader:
            images = images.to(DEVICE)
            text_feats = text_feats.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, text_feats)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(dataloader), correct / total


if __name__ == "__main__":
    train()

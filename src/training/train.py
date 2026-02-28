import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.fusion_model import MultimodalTriageModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "model_weights.pth")


def train():
    model = MultimodalTriageModel(num_classes=3).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for _ in range(5):
        images = torch.rand(8, 3, 224, 224).to(DEVICE)
        tokens = torch.randint(0, 10000, (8, 20)).to(DEVICE)
        labels = torch.randint(0, 3, (8,)).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, tokens)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print("Loss:", loss.item())

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("Saved weights to:", WEIGHTS_PATH)


if __name__ == "__main__":
    train()
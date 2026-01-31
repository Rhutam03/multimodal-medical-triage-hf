import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from app.fusion_model import MultimodalTriageModel
from app.training.real_dataset import RealMultimodalDataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LABELS = ["Low Risk", "Medium Risk", "High Risk"]

def evaluate():
    dataset = RealMultimodalDataset(
        labels_csv="data/labels.csv",
        image_dir="data/images"
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = MultimodalTriageModel(num_classes=3)
    checkpoint = torch.load("app/weights/triage_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, text_feats, labels in loader:
            images = images.to(DEVICE)
            text_feats = text_feats.to(DEVICE)

            outputs = model(images, text_feats)
            preds = outputs.argmax(dim=1).cpu()

            all_preds.extend(preds)
            all_labels.extend(labels)

    print("\n📊 Classification Report")
    print(classification_report(all_labels, all_preds, target_names=LABELS))

    print("🧩 Confusion Matrix")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()

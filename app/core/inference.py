import torch

from fusion_model import MultimodalTriageModel

DEVICE = torch.device("cpu")

model = MultimodalTriageModel(num_classes=3).to(DEVICE)
model.load_state_dict(
    torch.load("weights/model_weights.pth", map_location=DEVICE),
    strict=False
)
model.eval()


@torch.no_grad()
def predict_from_inputs(image, text):
    image = image.unsqueeze(0).to(DEVICE)
    logits = model(image, text)
    probs = torch.softmax(logits, dim=1)

    pred = probs.argmax(dim=1).item()
    conf = probs.max().item()
    return pred, conf

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights="IMAGENET1K_V2")
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(2048, 256)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, image: Image.Image):
        x = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.encoder(x).squeeze()
        return self.fc(features)

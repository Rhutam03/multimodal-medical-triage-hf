import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ImageEncoder(nn.Module):
    """
    Encodes a medical image into a fixed 256-dimensional feature vector
    using a pretrained ResNet-50 backbone.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet-50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        # Projection head: 2048 -> 256
        self.fc = nn.Linear(2048, 256)

        # Image preprocessing (ImageNet standard)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, image: Image.Image):
        """
        Args:
            image (PIL.Image): Input medical image

        Returns:
            torch.Tensor: 256-dim image embedding
        """

        if image is None:
            return None

        # Preprocess image
        x = self.preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)

        # Feature extraction (frozen backbone)
        with torch.no_grad():
            features = self.encoder(x)            # (1, 2048, 1, 1)
            features = features.view(1, -1)       # (1, 2048)

        # Project to 256-dim
        features = self.fc(features)              # (1, 256)

        return features.squeeze(0)                 # (256,)

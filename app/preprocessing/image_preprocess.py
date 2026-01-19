from torchvision import transforms

# ISIC-style preprocessing
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(image):
    """
    image: PIL.Image
    returns: torch.Tensor [1, 3, 224, 224]
    """
    if image is None:
        return None

    return _transform(image).unsqueeze(0)

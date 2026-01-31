from torchvision import transforms

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()   # -> [3, 224, 224]
])

def preprocess_image(image):
    """
    Returns a 3D tensor: [C, H, W]
    """
    return _transform(image)
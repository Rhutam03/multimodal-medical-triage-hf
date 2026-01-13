import torchvision.transforms as T

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def preprocess_image(image):
    """
    image: PIL.Image or None
    """
    if image is None:
        return None

    return _transform(image).unsqueeze(0)

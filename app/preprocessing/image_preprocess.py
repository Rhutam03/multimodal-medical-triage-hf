from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

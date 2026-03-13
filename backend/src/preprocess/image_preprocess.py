from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])
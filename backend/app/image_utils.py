import io

from PIL import Image, UnidentifiedImageError

from src.preprocess.image_preprocess import image_transform


class InvalidImageError(ValueError):
    """Raised when the uploaded image cannot be decoded."""


def load_image_tensor(image_bytes: bytes):
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Uploaded file is not a valid image") from exc

    return image_transform(pil_image).unsqueeze(0)

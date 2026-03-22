from io import BytesIO

from PIL import Image
import pytest

from app.image_utils import InvalidImageError, load_image_tensor


def test_load_image_tensor_rejects_invalid_images() -> None:
    with pytest.raises(InvalidImageError):
        load_image_tensor(b"not-an-image")


def test_load_image_tensor_accepts_valid_images() -> None:
    image = Image.new("RGB", (8, 8), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    tensor = load_image_tensor(buffer.getvalue())

    assert tuple(tensor.shape) == (1, 3, 160, 160)

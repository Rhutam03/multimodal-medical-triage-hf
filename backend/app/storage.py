import uuid
import boto3

from app.config import get_settings

settings = get_settings()
s3 = boto3.client("s3", region_name=settings.aws_region)


def upload_image_bytes(image_bytes: bytes, content_type: str) -> str:
    extension = "jpg"
    if content_type == "image/png":
        extension = "png"
    elif content_type == "image/webp":
        extension = "webp"

    key = f"uploads/{uuid.uuid4()}.{extension}"

    s3.put_object(
        Bucket=settings.uploads_bucket,
        Key=key,
        Body=image_bytes,
        ContentType=content_type,
    )

    return f"https://{settings.uploads_bucket}.s3.amazonaws.com/{key}"
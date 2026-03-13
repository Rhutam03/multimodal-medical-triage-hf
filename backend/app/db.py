from datetime import datetime, timezone

import boto3

from app.config import get_settings

settings = get_settings()
dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
table = dynamodb.Table(settings.predictions_table)


def save_prediction(item: dict) -> None:
    table.put_item(Item=item)


def list_predictions(limit: int = 20) -> list[dict]:
    response = table.scan(Limit=limit)
    return response.get("Items", [])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
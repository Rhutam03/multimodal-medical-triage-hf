from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    environment: str
    model_version: str


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    predicted_class: str
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    warnings: list[str]
    image_url: str | None = None
    duration_ms: float
    saved: bool


class PredictionRecord(BaseModel):
    request_id: str
    timestamp: str
    predicted_class: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    notes: str
    image_url: str | None = None
    model_version: str


class PredictionListResponse(BaseModel):
    items: list[PredictionRecord]
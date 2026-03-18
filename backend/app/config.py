from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "multimodal-medical-triage-api"
    environment: str = "dev"
    model_version: str = "v1"

    aws_region: str = "us-east-2"
    uploads_bucket: str = "change-me-uploads-bucket"
    predictions_table: str = "medical-triage-predictions"

    allowed_origins: str = "*"
    confidence_threshold: float = 0.70

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

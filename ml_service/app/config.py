"""
Configuration module for BYOAI ML Service.

Uses pydantic-settings to load configuration from environment variables
with the ML_ prefix. Provides sensible defaults for all settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables with ML_ prefix."""

    MODEL_NAME: str = "facebook/bart-large-mnli"
    DEVICE: str = "cpu"
    MAX_LENGTH: int = 512
    CONFIDENCE_THRESHOLD: float = 0.3
    LOG_LEVEL: str = "INFO"
    APP_NAME: str = "BYOAI ML Service"

    model_config = {
        "env_prefix": "ML_",
        "case_sensitive": True,
    }


settings = Settings()

"""Application configuration using pydantic-settings.

Loads settings from environment variables with the GATEWAY_ prefix.
Example: GATEWAY_ML_SERVICE_URL=http://localhost:8001
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Gateway service configuration.

    All settings can be overridden via environment variables
    prefixed with GATEWAY_ (e.g., GATEWAY_ML_SERVICE_URL).
    """

    ML_SERVICE_URL: str = "http://ml-service:8001"
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW: int = 60
    LOG_LEVEL: str = "INFO"
    APP_NAME: str = "BYOAI Gateway"

    model_config = {"env_prefix": "GATEWAY_"}


settings = Settings()

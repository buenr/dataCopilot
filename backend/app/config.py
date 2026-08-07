"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_origin: str = "http://localhost:5173,http://127.0.0.1:5173"
    session_ttl_minutes: int = 30
    session_volume_prefix: str = "dc"
    sandbox_image: str = "data-copilot-sandbox:latest"
    sandbox_network: str = "data-copilot-sandbox"
    sandbox_allow_egress: bool = False
    docker_socket: str = "unix:///var/run/docker.sock"
    llm_provider: str = "mock"
    anthropic_api_key: str = Field(default="", repr=False)
    openai_api_key: str = Field(default="", repr=False)
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-5.6"
    sessions_dir: str = "sessions"
    # Bundled datasets the "Try sample data" button loads into a session.
    sample_data_dir: str = "sample_data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

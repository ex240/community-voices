"""Environment-based settings. No model client is created here."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", repr=False)
    generation_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_path: str = "./data/chroma"
    lookback_days: int = Field(default=7, gt=0)
    max_threads: int = Field(default=50, gt=0)
    max_comments_per_thread: int = Field(default=12, gt=0)
    http_timeout_seconds: int = Field(default=15, gt=0)
    max_http_requests: int = Field(default=80, gt=0)
    max_ingest_seconds: int = Field(default=120, gt=0)
    max_chunk_chars: int = Field(default=1200, gt=0)
    report_path: str = "./data/latest_report.json"
    theme_count: int = Field(default=6, gt=0)
    theme_overview_threads: int = Field(default=40, gt=0)
    theme_snippet_chars: int = Field(default=180, gt=0)
    retrieval_per_theme: int = Field(default=8, gt=0)
    evidence_per_thread: int = Field(default=2, gt=0)
    generation_temperature: float = Field(default=0.2, ge=0)
    generation_timeout_seconds: int = Field(default=60, gt=0)
    baseline_report_path: str = "./data/latest_baseline_report.json"
    comparison_path: str = "./data/latest_comparison.json"

    @property
    def api_key_configured(self) -> bool:
        """True when a non-empty API key is present. Does not test the provider."""
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Return cached settings. Call cache_clear() in tests after env changes."""
    return Settings()

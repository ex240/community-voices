import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_allow_missing_api_key() -> None:
    settings = Settings(_env_file=None)
    assert settings.openai_api_key == ""
    assert settings.api_key_configured is False
    assert settings.lookback_days == 7
    assert settings.max_threads == 50
    assert settings.max_comments_per_thread == 12
    assert settings.http_timeout_seconds == 15
    assert settings.max_http_requests == 80
    assert settings.max_ingest_seconds == 120
    assert settings.max_chunk_chars == 1200


def test_report_generation_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPORT_PATH", raising=False)
    monkeypatch.delenv("BASELINE_REPORT_PATH", raising=False)
    monkeypatch.delenv("COMPARISON_PATH", raising=False)
    settings = Settings(_env_file=None)
    assert settings.report_path == "./data/latest_report.json"
    assert settings.baseline_report_path == "./data/latest_baseline_report.json"
    assert settings.comparison_path == "./data/latest_comparison.json"
    assert settings.theme_count == 6
    assert settings.retrieval_per_theme == 8
    assert settings.evidence_per_thread == 2


def test_api_key_configured_ignores_whitespace() -> None:
    settings = Settings(openai_api_key="   ", _env_file=None)
    assert settings.api_key_configured is False


def test_positive_limits_are_required() -> None:
    with pytest.raises(ValidationError):
        Settings(lookback_days=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(max_threads=-1, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(max_comments_per_thread=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(http_timeout_seconds=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(max_http_requests=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(max_ingest_seconds=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(max_chunk_chars=0, _env_file=None)

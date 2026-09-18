import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma-test"))
    monkeypatch.setenv("REPORT_PATH", str(tmp_path / "latest_report.json"))
    monkeypatch.setenv("BASELINE_REPORT_PATH", str(tmp_path / "latest_baseline_report.json"))
    monkeypatch.setenv("COMPARISON_PATH", str(tmp_path / "latest_comparison.json"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

from fastapi.testclient import TestClient

from app.main import app


def test_health_without_api_key() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "openai_api_key_configured": False}
    assert "sk-" not in response.text


def test_homepage_title_and_empty_states() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "Community Voices — Hacker News" in html
    assert "Community Voices report" in html
    assert "Stage 5 of 5" not in html
    assert "Stage " not in html
    assert "bounded seven-day sample" in html
    assert "Weekly report (RAG-powered)" in html
    assert "RAG versus no-RAG comparison" in html
    assert "collected-status" in html
    assert "python -m app.ingest" in html
    assert "python -m app.generate_report" in html
    assert "python -m app.compare_reports" in html
    assert "No report has been generated yet." in html
    assert "No comparison has been generated yet." in html
    assert "weekly-report" in html
    assert "comparison-view" in html


def test_report_endpoint_starts_empty() -> None:
    client = TestClient(app)
    response = client.get("/api/report")
    assert response.status_code == 200
    assert response.json() == {"status": "empty"}


def test_report_endpoint_returns_persisted_report() -> None:
    from app.config import get_settings
    from app.report import (
        GeneratedReport,
        Prediction,
        SourceRecord,
        ThemeSection,
        save_report,
    )

    settings = get_settings()
    save_report(
        settings.report_path,
        GeneratedReport(
            kind="rag",
            window_start="2026-09-10T17:00:00+00:00",
            window_end="2026-09-17T17:00:00+00:00",
            generated_at="2026-09-17T18:00:00+00:00",
            model="fake-writer",
            embedding_model="fake-embed",
            chunk_count=2,
            thread_count=1,
            comment_count=2,
            overview_story_ids=[10],
            themes=[],
            retrospective=[ThemeSection(title="Languages", body="People discussed Rust [S1].")],
            predictions=[
                Prediction(
                    claim="Rust tooling may stay in the conversation.",
                    rationale="The sampled comments already focus on it [S1].",
                    uncertainty="This is a prediction, not a measured outcome.",
                )
            ],
            limitation="Bounded sample only.",
            sources=[
                SourceRecord(
                    citation_id="S1",
                    chunk_id="hn-1-0",
                    comment_id=1,
                    story_id=10,
                    story_title="Rust",
                    source_url="https://news.ycombinator.com/item?id=1",
                    created_at="2026-09-11T00:00:00+00:00",
                    text="Rust is interesting",
                    distance=0.1,
                )
            ],
            unknown_citation_ids=[],
        ),
    )
    client = TestClient(app)
    response = client.get("/api/report")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["report"]["kind"] == "rag"
    assert body["report"]["retrospective"][0]["title"] == "Languages"
    assert body["report"]["sources"][0]["source_url"].endswith("id=1")


def test_comparison_endpoint_starts_empty() -> None:
    client = TestClient(app)
    response = client.get("/api/comparison")
    assert response.status_code == 200
    assert response.json() == {"status": "empty"}


def test_comparison_endpoint_returns_persisted_artifact() -> None:
    from app.compare import save_comparison
    from app.config import get_settings

    settings = get_settings()
    save_comparison(
        settings.comparison_path,
        {
            "kind": "rag-vs-no-rag",
            "window_start": "2026-09-10T17:00:00+00:00",
            "window_end": "2026-09-17T17:00:00+00:00",
            "model": "fake-writer",
            "rag_evidence_supplied": True,
            "baseline_evidence_supplied": False,
            "observations": [
                {
                    "dimension": "Evidence grounding",
                    "rag": "Source-backed.",
                    "baseline": "No evidence.",
                }
            ],
            "rag": {"metrics": {"latency_seconds": None, "total_tokens": None}},
            "baseline": {"metrics": {"latency_seconds": 0.4, "total_tokens": 12}},
        },
    )
    client = TestClient(app)
    response = client.get("/api/comparison")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["comparison"]["kind"] == "rag-vs-no-rag"
    assert body["comparison"]["baseline_evidence_supplied"] is False
    assert body["comparison"]["rag"]["metrics"]["latency_seconds"] is None
    assert body["comparison"]["baseline"]["metrics"]["total_tokens"] == 12


def test_corpus_endpoint_starts_empty() -> None:
    client = TestClient(app)
    response = client.get("/api/corpus")
    assert response.status_code == 200
    assert response.json() == {"chunk_count": 0, "status": "empty"}


def test_static_stylesheet_is_served() -> None:
    client = TestClient(app)
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert "font-family" in response.text


def test_metric_labels_describe_end_to_end_latency_and_exclude_embeddings() -> None:
    client = TestClient(app)
    response = client.get("/static/app.js")
    assert response.status_code == 200
    js = response.text
    assert "End-to-end report latency" in js
    assert "Baseline generation latency" in js
    assert "embedding usage is not included" in js

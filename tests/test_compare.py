import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.baseline import BaselineReport
from app.compare import build_comparison, metrics_from_payload, require_matching_rag
from app.compare_reports import run_compare_reports
from app.config import Settings
from app.report import Prediction, ThemeSection, save_report
from app.writer import TokenUsage

WINDOW_START = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 17, 17, 0, tzinfo=UTC)


class ScriptedWriter:
    model = "fake-writer"

    def __init__(self, replies: list[str], usage: TokenUsage | None = None) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self.last_usage = usage

    def complete(self, *, instructions: str, input_text: str) -> str:
        self.calls.append((instructions, input_text))
        if not self.replies:
            raise RuntimeError("writing model failed")
        return self.replies.pop(0)


def _rag_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "rag",
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "generated_at": WINDOW_END.isoformat(),
        "model": "fake-writer",
        "limitation": "Bounded sample only.",
        "retrospective": [{"title": "Rust", "body": "Sampled comments discussed lifetimes [S1]."}],
        "predictions": [
            {
                "claim": "Rust may stay in view.",
                "rationale": "The sampled comments already focus on it [S1].",
                "uncertainty": "This is a prediction.",
            }
        ],
        "sources": [
            {
                "citation_id": "S1",
                "comment_id": 1,
                "story_id": 10,
                "story_title": "Rust lifetimes",
                "source_url": "https://news.ycombinator.com/item?id=1",
                "created_at": WINDOW_START.isoformat(),
            }
        ],
        "latency_seconds": 1.25,
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
    }
    payload.update(overrides)
    return payload


def _baseline_report() -> BaselineReport:
    return BaselineReport(
        kind="no-rag",
        label="No-RAG baseline",
        window_start=WINDOW_START.isoformat(),
        window_end=WINDOW_END.isoformat(),
        generated_at=WINDOW_END.isoformat(),
        model="fake-writer",
        evidence_supplied=False,
        retrospective=[
            ThemeSection(
                title="Insufficient evidence",
                body="No current Hacker News source material was supplied.",
            )
        ],
        predictions=[
            Prediction(
                claim="Software tools may remain a topic.",
                rationale="That is a standing interest, not observed evidence.",
                uncertainty="This is a prediction.",
            )
        ],
        limitation="No current Hacker News source material was supplied.",
        latency_seconds=0.4,
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
    )


def test_require_matching_rag_rejects_window_and_model_mismatch() -> None:
    rag = _rag_payload()
    require_matching_rag(
        rag,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        model="fake-writer",
    )
    with pytest.raises(RuntimeError, match="does not match requested"):
        require_matching_rag(
            rag,
            window_start=WINDOW_START,
            window_end=datetime(2026, 9, 18, 17, 0, tzinfo=UTC),
            model="fake-writer",
        )
    with pytest.raises(RuntimeError, match="does not match configured"):
        require_matching_rag(
            rag,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            model="gpt-4.1-mini",
        )


def test_metrics_from_payload_preserve_nulls() -> None:
    assert metrics_from_payload({}) == {
        "latency_seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    assert metrics_from_payload(_rag_payload())["total_tokens"] == 140


def test_comparison_artifact_uses_real_outputs(tmp_path: Path) -> None:
    rag = _rag_payload()
    comparison = build_comparison(
        rag=rag,
        baseline=_baseline_report(),
        rag_path=str(tmp_path / "latest_report.json"),
        baseline_path=str(tmp_path / "latest_baseline_report.json"),
        manual_note="RAG named sampled threads; baseline admitted missing evidence.",
    )
    assert comparison["rag_evidence_supplied"] is True
    assert comparison["baseline_evidence_supplied"] is False
    assert comparison["rag"]["metrics"]["latency_seconds"] == 1.25
    assert comparison["baseline"]["metrics"]["total_tokens"] == 30
    assert "sources" not in comparison["baseline"]
    dimensions = {row["dimension"]: row for row in comparison["observations"]}
    assert "S1" in dimensions["Evidence grounding"]["rag"]
    assert "No community evidence" in dimensions["Evidence grounding"]["baseline"]
    assert "Rust lifetimes" in dimensions["Specificity"]["rag"]
    assert "disclosed" in dimensions["Unsupported claims"]["baseline"].lower()
    assert comparison["manual_note"].startswith("RAG named")


def test_unavailable_rag_metrics_stay_null(tmp_path: Path) -> None:
    rag = _rag_payload()
    for key in ("latency_seconds", "input_tokens", "output_tokens", "total_tokens"):
        rag.pop(key)
    comparison = build_comparison(
        rag=rag,
        baseline=_baseline_report(),
        rag_path="rag.json",
        baseline_path="baseline.json",
    )
    assert comparison["rag"]["metrics"]["latency_seconds"] is None
    assert comparison["rag"]["metrics"]["total_tokens"] is None
    assert comparison["baseline"]["metrics"]["latency_seconds"] == 0.4


def test_compare_command_persists_baseline_and_rejects_mismatch(tmp_path: Path) -> None:
    from app.report import GeneratedReport, SourceRecord

    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        baseline_report_path=str(tmp_path / "latest_baseline_report.json"),
        comparison_path=str(tmp_path / "latest_comparison.json"),
        openai_api_key="",
        generation_model="fake-writer",
        _env_file=None,
    )
    save_report(
        settings.report_path,
        GeneratedReport(
            kind="rag",
            window_start=WINDOW_START.isoformat(),
            window_end=WINDOW_END.isoformat(),
            generated_at=WINDOW_END.isoformat(),
            model="fake-writer",
            embedding_model="fake-embed",
            chunk_count=1,
            thread_count=1,
            comment_count=1,
            overview_story_ids=[10],
            themes=[],
            retrospective=[ThemeSection(title="Rust", body="Lifetimes [S1].")],
            predictions=[
                Prediction(claim="May continue.", rationale="[S1]", uncertainty="Uncertain.")
            ],
            limitation="Bounded sample.",
            sources=[
                SourceRecord(
                    citation_id="S1",
                    chunk_id="hn-1-0",
                    comment_id=1,
                    story_id=10,
                    story_title="Rust",
                    source_url="https://news.ycombinator.com/item?id=1",
                    created_at=WINDOW_START.isoformat(),
                    text="lifetimes matter",
                    distance=0.1,
                )
            ],
            unknown_citation_ids=[],
            latency_seconds=2.0,
            total_tokens=50,
        ),
    )
    writer = ScriptedWriter(
        [
            json.dumps(
                {
                    "retrospective": [
                        {
                            "title": "No current evidence",
                            "body": "No current Hacker News source material was supplied.",
                        }
                    ],
                    "predictions": [
                        {
                            "claim": "Tools may remain a topic.",
                            "rationale": "Standing interest only.",
                            "uncertainty": "Prediction.",
                        }
                    ],
                }
            )
        ],
        usage=TokenUsage(8, 4, 12),
    )
    comparison, path = run_compare_reports(
        settings=settings,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        writer=writer,
        manual_note="Checked citations against comments.",
    )
    assert Path(path).exists()
    assert Path(settings.baseline_report_path).exists()
    assert comparison["baseline"]["metrics"]["total_tokens"] == 12
    assert comparison["rag"]["metrics"]["latency_seconds"] == 2.0
    leaked = " ".join(text for _, text in writer.calls)
    assert "lifetimes matter" not in leaked
    assert "[S1]" not in leaked
    assert "news.ycombinator.com" not in leaked

    with pytest.raises(RuntimeError, match="does not match requested"):
        run_compare_reports(
            settings=settings,
            window_start=WINDOW_START,
            window_end=datetime(2026, 9, 18, 17, 0, tzinfo=UTC),
            writer=writer,
        )


def test_compare_requires_matching_writer_model(tmp_path: Path) -> None:
    from app.report import GeneratedReport

    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        openai_api_key="",
        generation_model="gpt-4.1-mini",
        _env_file=None,
    )
    save_report(
        settings.report_path,
        GeneratedReport(
            kind="rag",
            window_start=WINDOW_START.isoformat(),
            window_end=WINDOW_END.isoformat(),
            generated_at=WINDOW_END.isoformat(),
            model="gpt-4.1-mini",
            embedding_model="fake-embed",
            chunk_count=0,
            thread_count=0,
            comment_count=0,
            overview_story_ids=[],
            themes=[],
            retrospective=[ThemeSection(title="X", body="Y")],
            predictions=[],
            limitation="Bounded sample.",
            sources=[],
            unknown_citation_ids=[],
        ),
    )
    writer = ScriptedWriter([json.dumps({"retrospective": [], "predictions": []})])
    with pytest.raises(RuntimeError, match="does not match configured"):
        run_compare_reports(
            settings=settings,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            writer=writer,
        )

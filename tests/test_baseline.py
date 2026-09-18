import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.baseline import (
    BASELINE_INSTRUCTIONS,
    baseline_input,
    generate_baseline_report,
    load_baseline,
    save_baseline,
)
from app.compare import prompt_contains_leakage
from app.config import Settings
from app.report import REPORT_INSTRUCTIONS, REPORT_TASK
from app.writer import (
    RAG_LATENCY_LABEL,
    TokenUsage,
    format_run_metrics,
    usage_from_response,
)

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


def _baseline_json() -> str:
    return json.dumps(
        {
            "retrospective": [
                {
                    "title": "Insufficient current-week evidence",
                    "body": (
                        "No current Hacker News source material was supplied, "
                        "so this week's discussions cannot be verified here."
                    ),
                }
            ],
            "predictions": [
                {
                    "claim": "Hacker News may continue discussing software tools.",
                    "rationale": (
                        "That is a standing community interest, "
                        "not a verified current-week observation."
                    ),
                    "uncertainty": "This is a prediction, not a measured outcome.",
                }
            ],
        }
    )


def test_baseline_prompt_has_no_corpus_evidence() -> None:
    text = baseline_input(WINDOW_START, WINDOW_END)
    assert "Hacker News" in text
    assert WINDOW_START.isoformat() in text
    assert WINDOW_END.isoformat() in text
    assert "json" in text.lower()
    assert prompt_contains_leakage(text, BASELINE_INSTRUCTIONS) == []
    rag_prose = "Several sampled comments discussed Rust lifetimes [S1]."
    assert rag_prose not in text
    assert rag_prose not in BASELINE_INSTRUCTIONS


def test_both_arms_share_the_high_level_task() -> None:
    assert REPORT_TASK in REPORT_INSTRUCTIONS
    assert REPORT_TASK in BASELINE_INSTRUCTIONS
    assert "retrieved Hacker News evidence" in REPORT_INSTRUCTIONS
    assert "NOT been supplied current Hacker News source material" in BASELINE_INSTRUCTIONS
    assert "Cite only supplied IDs" not in BASELINE_INSTRUCTIONS
    assert "Do not pretend you verified" in BASELINE_INSTRUCTIONS
    assert "may continue" in REPORT_TASK
    assert "is likely to continue" in REPORT_TASK
    assert "could remain" in REPORT_TASK
    assert "will continue" in REPORT_TASK


def test_baseline_generation_and_persistence(tmp_path: Path) -> None:
    writer = ScriptedWriter([_baseline_json()], usage=TokenUsage(3, 5, 8))
    report = generate_baseline_report(
        writer=writer,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        now=WINDOW_END,
    )
    assert report.kind == "no-rag"
    assert report.label == "No-RAG baseline"
    assert report.evidence_supplied is False
    assert report.model == "fake-writer"
    assert report.latency_seconds is not None
    assert report.total_tokens == 8
    assert "no current" in report.retrospective[0].body.lower()
    instructions, payload = writer.calls[0]
    assert payload == baseline_input(WINDOW_START, WINDOW_END)
    assert REPORT_TASK in instructions
    path = save_baseline(str(tmp_path / "latest_baseline_report.json"), report)
    loaded = load_baseline(str(path))
    assert loaded is not None
    assert loaded["kind"] == "no-rag"
    assert "sources" not in loaded
    assert loaded["total_tokens"] == 8


def test_baseline_missing_api_key(tmp_path: Path) -> None:
    from app.compare_reports import run_compare_reports
    from app.report import GeneratedReport, Prediction, ThemeSection, save_report

    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        baseline_report_path=str(tmp_path / "baseline.json"),
        comparison_path=str(tmp_path / "comparison.json"),
        openai_api_key="",
        _env_file=None,
    )
    save_report(
        settings.report_path,
        GeneratedReport(
            kind="rag",
            window_start=WINDOW_START.isoformat(),
            window_end=WINDOW_END.isoformat(),
            generated_at=WINDOW_END.isoformat(),
            model=settings.generation_model,
            embedding_model="fake-embed",
            chunk_count=0,
            thread_count=0,
            comment_count=0,
            overview_story_ids=[],
            themes=[],
            retrospective=[ThemeSection(title="X", body="Y")],
            predictions=[Prediction(claim="Z", rationale="", uncertainty="")],
            limitation="Bounded sample.",
            sources=[],
            unknown_citation_ids=[],
        ),
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_compare_reports(
            settings=settings,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        )


def test_baseline_generation_failure() -> None:
    with pytest.raises(RuntimeError, match="writing model failed"):
        generate_baseline_report(
            writer=ScriptedWriter([]),
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        )


def test_usage_from_response_reads_sdk_fields() -> None:
    usage = usage_from_response(
        SimpleNamespace(usage=SimpleNamespace(input_tokens=11, output_tokens=4, total_tokens=15))
    )
    assert usage == TokenUsage(11, 4, 15)
    assert usage_from_response(SimpleNamespace(usage=None)) is None


def test_format_run_metrics_labels_end_to_end_latency_and_excludes_embeddings() -> None:
    text = format_run_metrics(
        latency_seconds=25.64,
        total_tokens=13716,
        latency_label=RAG_LATENCY_LABEL,
    )
    assert text.startswith("End-to-end report latency: 25.64s.")
    assert "Writing-model tokens only; embedding usage is not included." in text
    assert text.endswith("Total: 13716.")
    missing = format_run_metrics(
        latency_seconds=None,
        total_tokens=None,
        latency_label=RAG_LATENCY_LABEL,
    )
    assert "unavailable" in missing

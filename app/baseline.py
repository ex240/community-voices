"""No-RAG Community Voices baseline. This path must not receive corpus evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.report import REPORT_TASK, Prediction, ThemeSection, load_report, parse_written_report
from app.writer import TextGenerator, timed_complete

BASELINE_LIMITATION = (
    "No current Hacker News source material was supplied. This baseline was not "
    "given retrieved comments, titles, themes, corpus summaries, or search tools."
)
BASELINE_INSTRUCTIONS = (
    REPORT_TASK
    + """
You have NOT been supplied current Hacker News source material. You have no
retrieved comments, thread titles, corpus summaries, or tools.

Additional rules:
- Do not pretend you verified this week's discussions.
- Acknowledge the lack of current-week evidence where appropriate.
- If you cannot reliably identify that week's discussions without sources, say so.
- Do not fabricate citations, source links, counts, or measurements.
- Do not browse, search, or retrieve Hacker News content.
"""
)


@dataclass
class BaselineReport:
    kind: str
    label: str
    window_start: str
    window_end: str
    generated_at: str
    model: str
    evidence_supplied: bool
    retrospective: list[ThemeSection]
    predictions: list[Prediction]
    limitation: str
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def baseline_input(window_start: datetime, window_end: datetime) -> str:
    """Prompt payload: community, window, and task only. No corpus evidence."""
    return (
        "APPLICATION TASK: Write the weekly Community Voices JSON for Hacker News.\n"
        f"Report window [start, end): {window_start.isoformat()} -> {window_end.isoformat()}\n"
        "Community: Hacker News.\n"
        "Respond with a JSON object.\n\n"
        "No current Hacker News comments, titles, themes, or corpus summaries "
        "have been supplied. Do not browse or retrieve any."
    )


def generate_baseline_report(
    *,
    writer: TextGenerator,
    window_start: datetime,
    window_end: datetime,
    now: datetime | None = None,
) -> BaselineReport:
    input_text = baseline_input(window_start, window_end)
    text, latency_seconds, usage = timed_complete(
        writer,
        instructions=BASELINE_INSTRUCTIONS,
        input_text=input_text,
    )
    retrospective, predictions = parse_written_report(text)
    return BaselineReport(
        kind="no-rag",
        label="No-RAG baseline",
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        generated_at=(now or datetime.now(UTC)).isoformat(),
        model=writer.model,
        evidence_supplied=False,
        retrospective=retrospective,
        predictions=predictions,
        limitation=BASELINE_LIMITATION,
        latency_seconds=round(latency_seconds, 3),
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        total_tokens=None if usage is None else usage.total_tokens,
    )


def save_baseline(path: str, report: BaselineReport) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": report.kind,
        "label": report.label,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "generated_at": report.generated_at,
        "model": report.model,
        "evidence_supplied": report.evidence_supplied,
        "retrospective": [asdict(item) for item in report.retrospective],
        "predictions": [asdict(item) for item in report.predictions],
        "limitation": report.limitation,
        "latency_seconds": report.latency_seconds,
        "input_tokens": report.input_tokens,
        "output_tokens": report.output_tokens,
        "total_tokens": report.total_tokens,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_baseline(path: str) -> dict[str, object] | None:
    return load_report(path)

"""Compare a matching RAG report with a fresh no-RAG baseline."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from app.baseline import generate_baseline_report, save_baseline
from app.compare import build_comparison, require_matching_rag, save_comparison
from app.config import Settings, get_settings
from app.generate_report import resolve_report_window
from app.hn_client import parse_utc_datetime
from app.report import load_report
from app.store import ChunkStore
from app.writer import (
    BASELINE_LATENCY_LABEL,
    RAG_LATENCY_LABEL,
    OpenAIWriter,
    TextGenerator,
    format_run_metrics,
)


def run_compare_reports(
    *,
    settings: Settings | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    writer: TextGenerator | None = None,
    manual_note: str = "",
) -> tuple[dict[str, object], str]:
    settings = settings or get_settings()
    rag = load_report(settings.report_path)
    start, end = _window_for_comparison(
        settings=settings,
        rag=rag,
        window_start=window_start,
        window_end=window_end,
    )
    rag = require_matching_rag(
        rag,
        window_start=start,
        window_end=end,
        model=settings.generation_model,
    )
    if writer is None:
        if not settings.api_key_configured:
            raise RuntimeError(
                "Comparison needs OPENAI_API_KEY to generate the no-RAG baseline. "
                "Copy .env.example to .env and add a key."
            )
        writer = OpenAIWriter(
            settings.openai_api_key,
            settings.generation_model,
            temperature=settings.generation_temperature,
            timeout_seconds=settings.generation_timeout_seconds,
        )
    if writer.model != settings.generation_model:
        raise RuntimeError(
            f"Baseline writer model {writer.model} does not match configured "
            f"{settings.generation_model}."
        )
    baseline = generate_baseline_report(
        writer=writer,
        window_start=start,
        window_end=end,
    )
    baseline_path = str(save_baseline(settings.baseline_report_path, baseline))
    comparison = build_comparison(
        rag=rag,
        baseline=baseline,
        rag_path=settings.report_path,
        baseline_path=baseline_path,
        manual_note=manual_note,
    )
    path = str(save_comparison(settings.comparison_path, comparison))
    return comparison, path


def print_comparison_summary(comparison: dict[str, object], path: str) -> None:
    rag = comparison.get("rag") if isinstance(comparison.get("rag"), dict) else {}
    baseline = comparison.get("baseline") if isinstance(comparison.get("baseline"), dict) else {}
    print("Community Voices RAG vs no-RAG comparison")
    print(
        "UTC window [start, end): "
        f"{comparison.get('window_start')} -> {comparison.get('window_end')}"
    )
    print(f"Generation model: {comparison.get('model')}")
    print(f"RAG evidence supplied: {comparison.get('rag_evidence_supplied')}")
    print(f"Baseline evidence supplied: {comparison.get('baseline_evidence_supplied')}")
    print(_arm_metrics("RAG", rag.get("metrics") if isinstance(rag.get("metrics"), dict) else {}))
    print(
        _arm_metrics(
            "Baseline",
            baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {},
        )
    )
    print(f"Comparison written to: {path}")
    print(str(comparison.get("limitations") or ""))
    for row in comparison.get("observations") or []:
        if isinstance(row, dict):
            print(f"- {row.get('dimension')}: RAG={row.get('rag')} Baseline={row.get('baseline')}")


def _window_for_comparison(
    *,
    settings: Settings,
    rag: dict[str, object] | None,
    window_start: datetime | None,
    window_end: datetime | None,
) -> tuple[datetime, datetime]:
    if window_start is None and window_end is None:
        if rag is None:
            raise RuntimeError("No persisted RAG report. Run python -m app.generate_report first.")
        return (
            parse_utc_datetime(str(rag.get("window_start") or "")),
            parse_utc_datetime(str(rag.get("window_end") or "")),
        )
    with ChunkStore(settings.chroma_path) as store:
        return resolve_report_window(
            chroma_path=settings.chroma_path,
            chunks=store.list_chunks(),
            start=window_start,
            end=window_end,
        )


def _arm_metrics(label: str, metrics: dict[str, object]) -> str:
    latency_label = RAG_LATENCY_LABEL if label == "RAG" else BASELINE_LATENCY_LABEL
    return f"{label}: " + format_run_metrics(
        latency_seconds=_optional_float(metrics.get("latency_seconds")),
        total_tokens=_optional_int(metrics.get("total_tokens")),
        latency_label=latency_label,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a matching RAG report with a fresh no-RAG baseline."
    )
    parser.add_argument(
        "--start",
        help="UTC ISO start of the [start, end) window. Default: window from the RAG report.",
    )
    parser.add_argument(
        "--end",
        help="UTC ISO end of the [start, end) window. Default: window from the RAG report.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional manual comparison note stored with the artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        start = parse_utc_datetime(args.start) if args.start else None
        end = parse_utc_datetime(args.end) if args.end else None
        comparison, path = run_compare_reports(
            window_start=start,
            window_end=end,
            manual_note=args.note,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    print_comparison_summary(comparison, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

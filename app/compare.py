"""Compare a persisted RAG report with a fresh no-RAG baseline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.baseline import BaselineReport
from app.report import citation_ids_in

COMPARISON_LIMITATION = (
    "This comparison describes one window and one generation model. "
    "Latency and token counts are engineering metrics, not quality scores. "
    "It does not measure next-week prediction accuracy."
)
LEAKAGE_MARKERS = (
    "news.ycombinator.com",
    "retrieved comment text",
    "UNTRUSTED HACKER NEWS",
    "story_id=",
    "comment_id=",
    "chunk_id",
    "[S1]",
    "corpus overview",
    "bounded sample of Hacker News comments",
)


def require_matching_rag(
    rag: dict[str, object] | None,
    *,
    window_start: datetime,
    window_end: datetime,
    model: str,
) -> dict[str, object]:
    if rag is None:
        raise RuntimeError(
            "No persisted RAG report. Run python -m app.generate_report for this window first."
        )
    if rag.get("kind") != "rag":
        raise RuntimeError("The persisted report is not a RAG report.")
    expected_start = window_start.isoformat()
    expected_end = window_end.isoformat()
    actual_start = str(rag.get("window_start") or "")
    actual_end = str(rag.get("window_end") or "")
    if actual_start != expected_start or actual_end != expected_end:
        raise RuntimeError(
            "RAG report window "
            f"{actual_start} -> {actual_end} does not match requested "
            f"{expected_start} -> {expected_end}."
        )
    actual_model = str(rag.get("model") or "")
    if actual_model != model:
        raise RuntimeError(f"RAG report model {actual_model} does not match configured {model}.")
    return rag


def metrics_from_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "latency_seconds": payload.get("latency_seconds"),
        "input_tokens": payload.get("input_tokens"),
        "output_tokens": payload.get("output_tokens"),
        "total_tokens": payload.get("total_tokens"),
    }


def build_comparison(
    *,
    rag: dict[str, object],
    baseline: BaselineReport,
    rag_path: str,
    baseline_path: str,
    manual_note: str = "",
) -> dict[str, object]:
    rag_snapshot = _rag_snapshot(rag)
    baseline_snapshot = _baseline_snapshot(baseline)
    return {
        "kind": "rag-vs-no-rag",
        "window_start": rag_snapshot["window_start"],
        "window_end": rag_snapshot["window_end"],
        "model": rag_snapshot["model"],
        "generated_at": baseline.generated_at,
        "rag_evidence_supplied": True,
        "baseline_evidence_supplied": False,
        "rag_path": rag_path,
        "baseline_path": baseline_path,
        "rag": rag_snapshot,
        "baseline": baseline_snapshot,
        "observations": observe_outputs(rag, baseline_snapshot),
        "manual_note": manual_note.strip(),
        "limitations": COMPARISON_LIMITATION,
    }


def observe_outputs(
    rag: dict[str, object],
    baseline: dict[str, object],
) -> list[dict[str, str]]:
    rag_sources = rag.get("sources") if isinstance(rag.get("sources"), list) else []
    rag_ids = citation_ids_in(*_prose_from_payload(rag))
    baseline_ids = citation_ids_in(*_prose_from_payload(baseline))
    rag_titles = _story_titles(rag_sources)
    baseline_text = " ".join(_prose_from_payload(baseline)).lower()
    admitted_gap = any(
        phrase in baseline_text
        for phrase in (
            "not been supplied",
            "no current",
            "cannot reliably",
            "without source",
            "lack of",
            "no access",
            "not given",
        )
    )
    return [
        {
            "dimension": "Evidence grounding",
            "rag": (
                f"Source-backed: {len(rag_sources)} stored sources; "
                f"cited IDs in prose: {', '.join(rag_ids) or 'none'}."
            ),
            "baseline": "No community evidence was supplied. There is no Sources section.",
        },
        {
            "dimension": "Specificity",
            "rag": (
                "Names sampled threads such as: " + "; ".join(rag_titles[:5])
                if rag_titles
                else "No stored story titles."
            ),
            "baseline": (
                "Generic or prior-knowledge themes without sampled thread titles."
                if not _has_hn_item_links(baseline)
                else "Mentions Hacker News item links even though no sources were supplied."
            ),
        },
        {
            "dimension": "Verifiability",
            "rag": (
                "Reviewers can open stored Hacker News URLs for cited comments."
                if _has_hn_item_links(rag)
                else "No stored Hacker News URLs."
            ),
            "baseline": "Claims cannot be traced to supplied current-week comments.",
        },
        {
            "dimension": "Unsupported claims",
            "rag": "Retrospective claims are constrained to supplied evidence IDs.",
            "baseline": (
                "Baseline disclosed that current-week source material was not supplied."
                if admitted_gap
                else "Baseline made current-week claims without supplied evidence."
            ),
        },
        {
            "dimension": "Prediction rationale",
            "rag": (
                "Next-week predictions cite retrieved evidence."
                if any("[S" in text for text in _prediction_texts(rag))
                else "Predictions do not cite retrieved evidence IDs."
            ),
            "baseline": (
                "Predictions cannot be tied to observed current-week evidence."
                if not baseline_ids
                else "Baseline invented citation-like IDs without an evidence set."
            ),
        },
        {
            "dimension": "Coverage disclosure",
            "rag": str(rag.get("limitation") or "No limitation text stored."),
            "baseline": str(baseline.get("limitation") or "No limitation text stored."),
        },
    ]


def save_comparison(path: str, comparison: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return destination


def load_comparison(path: str) -> dict[str, object] | None:
    from app.report import load_report

    return load_report(path)


def prompt_contains_leakage(*texts: str) -> list[str]:
    found: list[str] = []
    blob = "\n".join(texts).lower()
    for marker in LEAKAGE_MARKERS:
        if marker.lower() in blob:
            found.append(marker)
    return found


def _rag_snapshot(rag: dict[str, object]) -> dict[str, object]:
    sources = rag.get("sources") if isinstance(rag.get("sources"), list) else []
    compact_sources = [
        {
            "citation_id": item.get("citation_id"),
            "comment_id": item.get("comment_id"),
            "story_id": item.get("story_id"),
            "story_title": item.get("story_title"),
            "source_url": item.get("source_url"),
            "created_at": item.get("created_at"),
        }
        for item in sources
        if isinstance(item, dict)
    ]
    return {
        "label": "RAG-powered report",
        "kind": rag.get("kind"),
        "window_start": rag.get("window_start"),
        "window_end": rag.get("window_end"),
        "generated_at": rag.get("generated_at"),
        "model": rag.get("model"),
        "evidence_supplied": True,
        "limitation": rag.get("limitation"),
        "retrospective": rag.get("retrospective") or [],
        "predictions": rag.get("predictions") or [],
        "sources": compact_sources,
        "metrics": metrics_from_payload(rag),
    }


def _baseline_snapshot(report: BaselineReport) -> dict[str, object]:
    from dataclasses import asdict

    return {
        "label": report.label,
        "kind": report.kind,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "generated_at": report.generated_at,
        "model": report.model,
        "evidence_supplied": False,
        "limitation": report.limitation,
        "retrospective": [asdict(item) for item in report.retrospective],
        "predictions": [asdict(item) for item in report.predictions],
        "metrics": {
            "latency_seconds": report.latency_seconds,
            "input_tokens": report.input_tokens,
            "output_tokens": report.output_tokens,
            "total_tokens": report.total_tokens,
        },
    }


def _prose_from_payload(payload: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for row in payload.get("retrospective") or []:
        if isinstance(row, dict):
            texts.extend([str(row.get("title") or ""), str(row.get("body") or "")])
    texts.extend(_prediction_texts(payload))
    return texts


def _prediction_texts(payload: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for row in payload.get("predictions") or []:
        if isinstance(row, dict):
            texts.extend(
                [
                    str(row.get("claim") or ""),
                    str(row.get("rationale") or ""),
                    str(row.get("uncertainty") or ""),
                ]
            )
    return texts


def _story_titles(sources: object) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    if not isinstance(sources, list):
        return titles
    for item in sources:
        if not isinstance(item, dict):
            continue
        title = str(item.get("story_title") or "").strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def _has_hn_item_links(payload: dict[str, object]) -> bool:
    blob = json.dumps(payload)
    return "news.ycombinator.com/item?id=" in blob

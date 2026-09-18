"""Bounded, stable overview of the current corpus for theme discovery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from app.hn_client import shorten
from app.store import RetrievedChunk


@dataclass(frozen=True)
class CorpusOverview:
    window_start: datetime
    window_end: datetime
    chunk_count: int
    thread_count: int
    comment_count: int
    story_ids: list[int]
    text: str


def build_corpus_overview(
    chunks: list[RetrievedChunk],
    *,
    window_start: datetime,
    window_end: datetime,
    max_threads: int,
    snippet_chars: int,
) -> CorpusOverview:
    """Summarize threads with titles, counts, and short snippets — not full comments."""
    grouped: dict[int, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        story_id = _meta_int(chunk.metadata.get("story_id"))
        if story_id is None:
            continue
        grouped[story_id].append(chunk)

    story_ids = sorted(grouped)
    selected_ids = story_ids[:max_threads]
    lines = [
        (
            "Bounded sample window [start, end): "
            f"{window_start.isoformat()} -> {window_end.isoformat()}"
        ),
        f"Chunks in window: {len(chunks)}",
        f"Threads in window: {len(story_ids)}",
        f"Threads included in this overview: {len(selected_ids)}",
        "Counts below are for this sample only. Do not treat them as site-wide popularity.",
        "",
    ]
    for story_id in selected_ids:
        thread_chunks = sorted(grouped[story_id], key=_comment_sort_key)
        title = str(thread_chunks[0].metadata.get("story_title") or "Untitled")
        thread_comment_ids = {
            cid
            for chunk in thread_chunks
            if (cid := _meta_int(chunk.metadata.get("comment_id"))) is not None
        }
        lines.append(
            f"- story_id={story_id} title={title} comments_in_sample={len(thread_comment_ids)}"
        )
        for chunk in thread_chunks[:2]:
            snippet = shorten(chunk.document, snippet_chars)
            comment_id = _meta_int(chunk.metadata.get("comment_id"))
            lines.append(f"    comment_id={comment_id} {snippet}")
        lines.append("")

    return CorpusOverview(
        window_start=window_start,
        window_end=window_end,
        chunk_count=len(chunks),
        thread_count=len(story_ids),
        comment_count=len(
            {
                cid
                for chunk in chunks
                if (cid := _meta_int(chunk.metadata.get("comment_id"))) is not None
            }
        ),
        story_ids=selected_ids,
        text="\n".join(lines).strip(),
    )


def _meta_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _comment_sort_key(chunk: RetrievedChunk) -> tuple[int, str]:
    comment_id = _meta_int(chunk.metadata.get("comment_id")) or 0
    return (comment_id, chunk.chunk_id)

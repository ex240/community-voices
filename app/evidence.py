"""Theme-specific vector retrieval with a per-thread evidence cap."""

from __future__ import annotations

from dataclasses import dataclass

from app.embedder import Embedder
from app.store import ChunkStore, RetrievedChunk


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    comment_id: int
    story_id: int
    story_title: str
    source_url: str
    created_at: str
    text: str
    distance: float


def retrieve_theme_evidence(
    store: ChunkStore,
    embedder: Embedder,
    *,
    query: str,
    start_ts: int,
    end_ts: int,
    fetch_count: int,
    keep_count: int,
    per_thread: int,
) -> list[EvidenceItem]:
    """Retrieve TEXT near the theme query. Distance is nearness only, not confidence."""
    if not query.strip():
        return []
    vector = embedder.embed_texts([query])[0]
    hits = store.query(
        vector,
        n_results=max(fetch_count, keep_count),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    return select_evidence(hits, keep_count=keep_count, per_thread=per_thread)


def select_evidence(
    hits: list[RetrievedChunk],
    *,
    keep_count: int,
    per_thread: int,
) -> list[EvidenceItem]:
    """Dedupe IDs, cap evidence per thread, keep nearer unique chunks."""
    selected: list[EvidenceItem] = []
    seen_chunks: set[str] = set()
    seen_comments: set[int] = set()
    per_story: dict[int, int] = {}
    ranked = sorted(hits, key=lambda hit: (hit.distance, hit.chunk_id))
    for hit in ranked:
        if len(selected) >= keep_count:
            break
        item = evidence_from_hit(hit)
        if item is None:
            continue
        if item.chunk_id in seen_chunks or item.comment_id in seen_comments:
            continue
        used = per_story.get(item.story_id, 0)
        if used >= per_thread:
            continue
        seen_chunks.add(item.chunk_id)
        seen_comments.add(item.comment_id)
        per_story[item.story_id] = used + 1
        selected.append(item)
    return selected


def evidence_from_hit(hit: RetrievedChunk) -> EvidenceItem | None:
    comment_id = _meta_int(hit.metadata.get("comment_id"))
    story_id = _meta_int(hit.metadata.get("story_id"))
    text = (hit.document or "").strip()
    if comment_id is None or story_id is None or not text:
        return None
    return EvidenceItem(
        chunk_id=hit.chunk_id,
        comment_id=comment_id,
        story_id=story_id,
        story_title=str(hit.metadata.get("story_title") or "Untitled"),
        source_url=str(hit.metadata.get("source_url") or ""),
        created_at=str(hit.metadata.get("created_at") or ""),
        text=text,
        distance=float(hit.distance),
    )


def _meta_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

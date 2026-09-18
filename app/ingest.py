"""Bounded Hacker News → Chroma ingestion. No report generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.chunking import PreparedChunk, chunks_from_comment
from app.config import Settings, get_settings
from app.embedder import Embedder, OpenAIEmbedder
from app.hn_client import (
    DEFAULT_USER_AGENT,
    HackerNewsClient,
    HackerNewsError,
    RequestBudgetError,
    classify_comment_hit,
    parse_utc_datetime,
    resolve_ingest_window,
    round_robin_ids,
    select_hits_by_id,
    split_window,
    story_ids_from_hits,
)
from app.store import ChunkStore

SAMPLING_NOTE = (
    "This is a bounded sample: one discovery request per lookback day, then at most "
    "MAX_THREADS threads and MAX_COMMENTS_PER_THREAD comments per thread, chosen by "
    "lowest source IDs so reruns of the same window stay stable. "
    "Each ingest replaces the stored corpus with that sample."
)
DISCOVERY_HITS_PER_BUCKET = 1000
COMMENT_CANDIDATE_HITS = 1000


@dataclass
class SampleManifest:
    window_start: str
    window_end: str
    story_ids: list[int]
    comment_ids: list[int]
    chunk_ids: list[str]


@dataclass
class IngestStats:
    window_start: datetime
    window_end: datetime
    threads_considered: int = 0
    comments_fetched: int = 0
    comments_skipped_empty: int = 0
    comments_skipped_deleted: int = 0
    comments_skipped_out_of_window: int = 0
    chunks_new: int = 0
    chunks_changed: int = 0
    chunks_unchanged: int = 0
    chunks_removed: int = 0
    embeddings_created: int = 0
    failed_requests: int = 0
    requests_used: int = 0
    corpus_size: int = 0
    stopped_reason: str | None = None
    sampling_note: str = SAMPLING_NOTE
    story_ids: list[int] = field(default_factory=list)
    comment_ids: list[int] = field(default_factory=list)
    previous_overlap: str | None = None
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def chunks_written(self) -> int:
        return self.chunks_new + self.chunks_changed

    def print_report(self) -> None:
        print("Community Voices ingestion")
        print(
            "UTC window [start, end): "
            f"{self.window_start.isoformat()} -> {self.window_end.isoformat()}"
        )
        print("Reuse this window with:")
        print(
            "  python -m app.ingest "
            f"--start {self.window_start.isoformat()} --end {self.window_end.isoformat()}"
        )
        print(f"Threads considered: {self.threads_considered}")
        print(f"Story IDs: {_compact_ids(self.story_ids)}")
        print(f"Comments fetched: {self.comments_fetched}")
        print(f"Comment IDs: {_compact_ids(self.comment_ids)}")
        print(f"Skipped empty: {self.comments_skipped_empty}")
        print(f"Skipped deleted: {self.comments_skipped_deleted}")
        print(f"Skipped out of window: {self.comments_skipped_out_of_window}")
        print(f"Chunks new: {self.chunks_new}")
        print(f"Chunks changed: {self.chunks_changed}")
        print(f"Chunks unchanged (hash match, not re-embedded): {self.chunks_unchanged}")
        print(f"Chunks written (new + changed): {self.chunks_written}")
        print(f"Obsolete / out-of-sample chunks removed: {self.chunks_removed}")
        print(f"Embeddings created: {self.embeddings_created}")
        print(f"Corpus size after ingest: {self.corpus_size}")
        print(f"Failed requests: {self.failed_requests}")
        print(f"HN requests used: {self.requests_used}")
        if self.previous_overlap:
            print(self.previous_overlap)
        if self.stopped_reason:
            print(f"Stopped early: {self.stopped_reason}")
        print(f"Sampling: {self.sampling_note}")


def ingest_window(
    *,
    hn: HackerNewsClient,
    store: ChunkStore,
    embedder: Embedder,
    start: datetime,
    end: datetime,
    max_threads: int,
    max_comments_per_thread: int,
    max_chunk_chars: int,
    bucket_count: int,
    deadline: float,
    sample_path: Path | None = None,
) -> IngestStats:
    """Fetch a bounded sample, chunk it, and upsert into Chroma."""
    stats = IngestStats(window_start=start, window_end=end)
    story_ids = _discover_story_ids(
        hn,
        start,
        end,
        max_threads=max_threads,
        bucket_count=bucket_count,
        stats=stats,
        deadline=deadline,
    )
    stats.story_ids = story_ids
    stats.threads_considered = len(story_ids)

    pending: list[PreparedChunk] = []
    sample_chunk_ids: set[str] = set()
    for story_id in story_ids:
        if _timed_out(deadline, stats):
            break
        hits = _fetch_story_comments(
            hn,
            start,
            end,
            story_id=story_id,
            max_comments=max_comments_per_thread,
            stats=stats,
        )
        if hits is None:
            continue
        pending.extend(
            _process_story_hits(
                hits,
                start,
                end,
                store=store,
                max_chunk_chars=max_chunk_chars,
                stats=stats,
                sample_chunk_ids=sample_chunk_ids,
            )
        )

    _write_changed_chunks(pending, store, embedder, stats)
    if not stats.stopped_reason:
        stats.chunks_removed += _drop_out_of_sample(store, sample_chunk_ids)
    stats.chunk_ids = sorted(sample_chunk_ids)
    stats.corpus_size = store.count()
    stats.requests_used = hn.requests_used
    _record_sample(stats, sample_path)
    return stats


def _discover_story_ids(
    hn: HackerNewsClient,
    start: datetime,
    end: datetime,
    *,
    max_threads: int,
    bucket_count: int,
    stats: IngestStats,
    deadline: float,
) -> list[int]:
    groups: list[list[int]] = []
    buckets = split_window(start, end, parts=bucket_count)
    for bucket_start, bucket_end in buckets:
        if _timed_out(deadline, stats):
            break
        try:
            hits = hn.search_comments(
                bucket_start,
                bucket_end,
                hits_per_page=DISCOVERY_HITS_PER_BUCKET,
            )
            groups.append(sorted(story_ids_from_hits(hits)))
        except RequestBudgetError:
            stats.stopped_reason = stats.stopped_reason or "request budget"
            break
        except HackerNewsError:
            stats.failed_requests += 1
            groups.append([])
    return round_robin_ids(groups, max_threads) if groups else []


def _fetch_story_comments(
    hn: HackerNewsClient,
    start: datetime,
    end: datetime,
    *,
    story_id: int,
    max_comments: int,
    stats: IngestStats,
) -> list[dict] | None:
    try:
        hits = hn.search_comments(
            start,
            end,
            hits_per_page=COMMENT_CANDIDATE_HITS,
            story_id=story_id,
        )
    except RequestBudgetError:
        stats.stopped_reason = stats.stopped_reason or "request budget"
        return None
    except HackerNewsError:
        stats.failed_requests += 1
        return None
    selected = select_hits_by_id(hits, max_comments)
    stats.comments_fetched += len(selected)
    return selected


def _process_story_hits(
    hits: list[dict],
    start: datetime,
    end: datetime,
    *,
    store: ChunkStore,
    max_chunk_chars: int,
    stats: IngestStats,
    sample_chunk_ids: set[str],
) -> list[PreparedChunk]:
    to_embed: list[PreparedChunk] = []
    for hit in hits:
        classified = classify_comment_hit(hit, start, end)
        if classified.status == "empty":
            stats.comments_skipped_empty += 1
            stats.chunks_removed += _delete_comment_chunks(store, classified.comment_id)
            continue
        if classified.status == "deleted":
            stats.comments_skipped_deleted += 1
            stats.chunks_removed += _delete_comment_chunks(store, classified.comment_id)
            continue
        if classified.status == "out_of_window":
            stats.comments_skipped_out_of_window += 1
            continue
        if classified.status != "ok" or classified.comment is None:
            stats.comments_skipped_empty += 1
            continue

        stats.comment_ids.append(classified.comment.comment_id)
        prepared = chunks_from_comment(classified.comment, max_chunk_chars)
        current_ids = {chunk.chunk_id for chunk in prepared}
        sample_chunk_ids.update(current_ids)
        existing_ids = store.ids_for_comment(classified.comment.comment_id)
        stale_ids = [item for item in existing_ids if item not in current_ids]
        if stale_ids:
            store.delete_ids(stale_ids)
            stats.chunks_removed += len(stale_ids)

        existing = store.get_metadata([chunk.chunk_id for chunk in prepared])
        for chunk in prepared:
            stored = existing.get(chunk.chunk_id)
            if stored and stored.get("content_hash") == chunk.content_hash:
                stats.chunks_unchanged += 1
            elif stored:
                stats.chunks_changed += 1
                to_embed.append(chunk)
            else:
                stats.chunks_new += 1
                to_embed.append(chunk)
    return to_embed


def _delete_comment_chunks(store: ChunkStore, comment_id: int | None) -> int:
    if comment_id is None:
        return 0
    ids = store.ids_for_comment(comment_id)
    if ids:
        store.delete_ids(ids)
    return len(ids)


def _drop_out_of_sample(store: ChunkStore, sample_chunk_ids: set[str]) -> int:
    extra = [chunk_id for chunk_id in store.list_ids() if chunk_id not in sample_chunk_ids]
    if extra:
        store.delete_ids(extra)
    return len(extra)


def _write_changed_chunks(
    chunks: list[PreparedChunk],
    store: ChunkStore,
    embedder: Embedder,
    stats: IngestStats,
) -> None:
    if not chunks:
        return
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    store.upsert_chunks(chunks, vectors, embedder.model)
    stats.embeddings_created = len(vectors)


def _timed_out(deadline: float, stats: IngestStats) -> bool:
    if time.monotonic() >= deadline:
        stats.stopped_reason = stats.stopped_reason or "time limit"
        return True
    return False


def _sample_manifest_path(chroma_path: str) -> Path:
    return Path(chroma_path).expanduser().resolve().parent / "last_ingest_sample.json"


def load_ingest_manifest(chroma_path: str) -> SampleManifest | None:
    """Return the last successful ingest sample, if one was recorded."""
    return _load_manifest(_sample_manifest_path(chroma_path))


def _record_sample(stats: IngestStats, sample_path: Path | None) -> None:
    if sample_path is None:
        return
    current = SampleManifest(
        window_start=stats.window_start.isoformat(),
        window_end=stats.window_end.isoformat(),
        story_ids=list(stats.story_ids),
        comment_ids=list(stats.comment_ids),
        chunk_ids=list(stats.chunk_ids),
    )
    previous = _load_manifest(sample_path)
    same_window = (
        previous is not None
        and previous.window_start == current.window_start
        and previous.window_end == current.window_end
    )
    if previous is not None and same_window:
        stats.previous_overlap = _overlap_summary(previous, current)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(asdict(current), indent=2) + "\n", encoding="utf-8")


def _load_manifest(path: Path) -> SampleManifest | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SampleManifest(
            window_start=str(raw["window_start"]),
            window_end=str(raw["window_end"]),
            story_ids=[int(item) for item in raw.get("story_ids", [])],
            comment_ids=[int(item) for item in raw.get("comment_ids", [])],
            chunk_ids=[str(item) for item in raw.get("chunk_ids", [])],
        )
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _overlap_summary(previous: SampleManifest, current: SampleManifest) -> str:
    story_overlap = _overlap_counts(previous.story_ids, current.story_ids)
    comment_overlap = _overlap_counts(previous.comment_ids, current.comment_ids)
    chunk_overlap = _overlap_counts(previous.chunk_ids, current.chunk_ids)
    return (
        "Overlap with previous same-window run: "
        f"stories {story_overlap}; comments {comment_overlap}; chunks {chunk_overlap}."
    )


def _overlap_counts(previous: list[int] | list[str], current: list[int] | list[str]) -> str:
    prev = set(previous)
    curr = set(current)
    shared = len(prev & curr)
    return f"{shared}/{len(curr)} current also in previous ({len(prev)} previous)"


def _compact_ids(values: list[int], limit: int = 12) -> str:
    if not values:
        return "(none)"
    shown = ", ".join(str(item) for item in values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return shown


def run_ingest(
    settings: Settings | None = None,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> IngestStats:
    settings = settings or get_settings()
    if not settings.api_key_configured:
        raise RuntimeError(
            "Ingest needs OPENAI_API_KEY to create embeddings. "
            "Copy .env.example to .env and add a key."
        )
    start, end = resolve_ingest_window(
        lookback_days=settings.lookback_days,
        now=datetime.now(UTC),
        start=window_start,
        end=window_end,
    )
    deadline = time.monotonic() + settings.max_ingest_seconds
    timeout = httpx.Timeout(settings.http_timeout_seconds)
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    sample_path = _sample_manifest_path(settings.chroma_path)
    with (
        httpx.Client(timeout=timeout, headers=headers) as http,
        ChunkStore(settings.chroma_path) as store,
    ):
        hn = HackerNewsClient(http, max_requests=settings.max_http_requests)
        return ingest_window(
            hn=hn,
            store=store,
            embedder=embedder,
            start=start,
            end=end,
            max_threads=settings.max_threads,
            max_comments_per_thread=settings.max_comments_per_thread,
            max_chunk_chars=settings.max_chunk_chars,
            bucket_count=settings.lookback_days,
            deadline=deadline,
            sample_path=sample_path,
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a bounded HN sample into Chroma.")
    parser.add_argument(
        "--start",
        help="UTC ISO start of the [start, end) window. Default: end minus LOOKBACK_DAYS.",
    )
    parser.add_argument(
        "--end",
        help="UTC ISO end of the [start, end) window. Default: now. Pin this to test reruns.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        start = parse_utc_datetime(args.start) if args.start else None
        end = parse_utc_datetime(args.end) if args.end else None
        stats = run_ingest(window_start=start, window_end=end)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except HackerNewsError as exc:
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1
    stats.print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

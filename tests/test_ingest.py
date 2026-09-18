import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from app.hn_client import HackerNewsClient
from app.ingest import _parse_args, ingest_window
from app.store import ChunkStore

WINDOW_START = datetime(2026, 9, 9, 18, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 11, 18, 0, tzinfo=UTC)
MID = WINDOW_START + timedelta(days=1)


class RecordingEmbedder:
    model = "synthetic-test"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(item)), 1.0, 0.0] for item in texts]


def _hit(
    comment_id: int,
    story_id: int,
    created_at: datetime,
    text: str | None,
    title: str = "Thread",
) -> dict[str, object]:
    return {
        "objectID": str(comment_id),
        "story_id": story_id,
        "story_title": title,
        "created_at_i": int(created_at.timestamp()),
        "comment_text": text,
        "author": "tester",
    }


def _json(hits: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"hits": hits})


def _run_ingest(
    tmp_path: Path,
    handler,
    embedder: RecordingEmbedder,
    max_threads: int = 2,
    max_comments_per_thread: int = 3,
    sample_path: Path | None = None,
):
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http, ChunkStore(str(tmp_path / "chroma")) as store:
        hn = HackerNewsClient(http, max_requests=20)
        stats = ingest_window(
            hn=hn,
            store=store,
            embedder=embedder,
            start=WINDOW_START,
            end=WINDOW_END,
            max_threads=max_threads,
            max_comments_per_thread=max_comments_per_thread,
            max_chunk_chars=200,
            bucket_count=2,
            deadline=time.monotonic() + 30,
            sample_path=sample_path,
        )
        return stats, store.count()


def test_ingest_skips_empty_deleted_and_out_of_window(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        if tags == "comment":
            return _json([_hit(1, 101, WINDOW_START + timedelta(hours=1), "<p>Keep me</p>")])
        if tags == "comment,story_101":
            return _json(
                [
                    _hit(1, 101, WINDOW_START + timedelta(hours=1), "<p>Keep me</p>"),
                    _hit(2, 101, WINDOW_START + timedelta(hours=2), "<p>   </p>"),
                    _hit(3, 101, WINDOW_START + timedelta(hours=3), None),
                    _hit(4, 101, WINDOW_END, "<p>Too late</p>"),
                ]
            )
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    stats, count = _run_ingest(
        tmp_path,
        handler,
        embedder,
        max_threads=1,
        max_comments_per_thread=4,
    )
    assert stats.comments_skipped_empty == 1
    assert stats.comments_skipped_deleted == 1
    assert stats.comments_skipped_out_of_window == 1
    assert stats.chunks_written == 1
    assert stats.chunks_new == 1
    assert stats.chunks_changed == 0
    assert count == 1
    assert embedder.texts == ["Keep me"]


def test_ingest_is_idempotent_when_content_is_unchanged(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        if tags == "comment":
            return _json([_hit(10, 201, WINDOW_START + timedelta(hours=2), "<p>Stable text</p>")])
        if tags == "comment,story_201":
            return _json([_hit(10, 201, WINDOW_START + timedelta(hours=2), "<p>Stable text</p>")])
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    first, count_after_first = _run_ingest(tmp_path, handler, embedder, max_threads=1)
    second, count_after_second = _run_ingest(tmp_path, handler, embedder, max_threads=1)
    assert first.chunks_written == 1
    assert first.chunks_new == 1
    assert first.chunks_changed == 0
    assert second.chunks_written == 0
    assert second.chunks_new == 0
    assert second.chunks_changed == 0
    assert second.chunks_unchanged == 1
    assert second.embeddings_created == 0
    assert count_after_first == count_after_second == 1
    assert embedder.texts == ["Stable text"]


def test_changed_content_is_reembedded_without_duplicate_ids(tmp_path: Path) -> None:
    payload = {"text": "<p>Version one</p>"}

    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        if tags == "comment":
            return _json([_hit(11, 301, WINDOW_START + timedelta(hours=2), payload["text"])])
        if tags == "comment,story_301":
            return _json([_hit(11, 301, WINDOW_START + timedelta(hours=2), payload["text"])])
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    _run_ingest(tmp_path, handler, embedder, max_threads=1)
    payload["text"] = "<p>Version two</p>"
    stats, count = _run_ingest(tmp_path, handler, embedder, max_threads=1)
    assert stats.chunks_written == 1
    assert stats.chunks_new == 0
    assert stats.chunks_changed == 1
    assert stats.chunks_unchanged == 0
    assert count == 1
    assert embedder.texts == ["Version one", "Version two"]


def test_partial_request_failure_still_stores_successful_threads(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        filters = request.url.params.get("numericFilters", "")
        if tags == "comment":
            if f"created_at_i>={int(WINDOW_START.timestamp())}," in filters and (
                f"created_at_i<{int(MID.timestamp())}" in filters
            ):
                return _json([_hit(21, 401, WINDOW_START + timedelta(hours=1), "<p>Day one</p>")])
            return _json([_hit(22, 402, MID + timedelta(hours=1), "<p>Day two</p>")])
        if tags == "comment,story_401":
            return _json([_hit(21, 401, WINDOW_START + timedelta(hours=1), "<p>Day one</p>")])
        if tags == "comment,story_402":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    stats, count = _run_ingest(tmp_path, handler, embedder, max_threads=2)
    assert stats.failed_requests == 1
    assert stats.chunks_written == 1
    assert count == 1
    assert "Day one" in embedder.texts


def test_deleted_comment_removes_obsolete_chunks(tmp_path: Path) -> None:
    payload = {"text": "<p>Soon gone</p>"}

    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        if tags == "comment":
            discovery_text = payload["text"] or "<p>x</p>"
            return _json([_hit(31, 501, WINDOW_START + timedelta(hours=1), discovery_text)])
        if tags == "comment,story_501":
            return _json([_hit(31, 501, WINDOW_START + timedelta(hours=1), payload["text"])])
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    _run_ingest(tmp_path, handler, embedder, max_threads=1)
    payload["text"] = None
    stats, count = _run_ingest(tmp_path, handler, embedder, max_threads=1)
    assert stats.comments_skipped_deleted == 1
    assert stats.chunks_removed == 1
    assert count == 0


def test_comment_sample_keeps_lowest_ids_not_newest(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        if tags == "comment":
            return _json([_hit(30, 201, WINDOW_START + timedelta(hours=3), "<p>newest</p>")])
        if tags == "comment,story_201":
            return _json(
                [
                    _hit(30, 201, WINDOW_START + timedelta(hours=3), "<p>newest</p>"),
                    _hit(20, 201, WINDOW_START + timedelta(hours=2), "<p>middle</p>"),
                    _hit(10, 201, WINDOW_START + timedelta(hours=1), "<p>oldest</p>"),
                ]
            )
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    stats, count = _run_ingest(
        tmp_path,
        handler,
        embedder,
        max_threads=1,
        max_comments_per_thread=2,
    )
    assert stats.comment_ids == [10, 20]
    assert embedder.texts == ["oldest", "middle"]
    assert count == 2


def test_story_sample_is_stable_after_sorting_ids(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        filters = request.url.params.get("numericFilters", "")
        if tags == "comment":
            if f"created_at_i<{int(MID.timestamp())}" in filters:
                return _json(
                    [
                        _hit(3, 30, WINDOW_START + timedelta(hours=2), "<p>late story</p>"),
                        _hit(1, 10, WINDOW_START + timedelta(hours=1), "<p>early story</p>"),
                    ]
                )
            return _json([_hit(2, 20, MID + timedelta(hours=1), "<p>mid story</p>")])
        if tags == "comment,story_10":
            return _json([_hit(1, 10, WINDOW_START + timedelta(hours=1), "<p>early story</p>")])
        if tags == "comment,story_20":
            return _json([_hit(2, 20, MID + timedelta(hours=1), "<p>mid story</p>")])
        if tags == "comment,story_30":
            return _json([_hit(3, 30, WINDOW_START + timedelta(hours=2), "<p>late story</p>")])
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    stats, _count = _run_ingest(tmp_path, handler, embedder, max_threads=2)
    assert stats.story_ids == [10, 20]


def test_out_of_sample_chunks_are_removed(tmp_path: Path) -> None:
    state = {"story": 201, "comment": 10, "text": "<p>First sample</p>"}

    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        story = state["story"]
        comment = state["comment"]
        if tags == "comment":
            return _json([_hit(comment, story, WINDOW_START + timedelta(hours=1), state["text"])])
        if tags == f"comment,story_{story}":
            return _json([_hit(comment, story, WINDOW_START + timedelta(hours=1), state["text"])])
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    first, count_after_first = _run_ingest(tmp_path, handler, embedder, max_threads=1)
    state.update({"story": 401, "comment": 40, "text": "<p>Second sample</p>"})
    second, count_after_second = _run_ingest(tmp_path, handler, embedder, max_threads=1)
    assert count_after_first == 1
    assert count_after_second == 1
    assert first.chunks_new == 1
    assert second.chunks_new == 1
    assert second.chunks_removed == 1
    assert embedder.texts == ["First sample", "Second sample"]


def test_same_window_overlap_is_reported(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tags = request.url.params.get("tags")
        if tags == "comment":
            return _json([_hit(10, 201, WINDOW_START + timedelta(hours=2), "<p>Stable</p>")])
        if tags == "comment,story_201":
            return _json([_hit(10, 201, WINDOW_START + timedelta(hours=2), "<p>Stable</p>")])
        return httpx.Response(400, json={"message": "unexpected"})

    embedder = RecordingEmbedder()
    sample_path = tmp_path / "last_ingest_sample.json"
    _run_ingest(tmp_path, handler, embedder, max_threads=1, sample_path=sample_path)
    second, _count = _run_ingest(
        tmp_path, handler, embedder, max_threads=1, sample_path=sample_path
    )
    assert second.previous_overlap is not None
    assert "stories 1/1 current also in previous" in second.previous_overlap
    assert second.chunks_new == 0
    assert second.chunks_changed == 0


def test_cli_accepts_fixed_window_flags() -> None:
    args = _parse_args(["--start", "2026-09-10T17:00:00Z", "--end", "2026-09-17T17:00:00Z"])
    assert args.start == "2026-09-10T17:00:00Z"
    assert args.end == "2026-09-17T17:00:00Z"

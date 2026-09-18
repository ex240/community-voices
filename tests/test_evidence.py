from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.chunking import PreparedChunk, content_hash
from app.evidence import retrieve_theme_evidence, select_evidence
from app.store import ChunkStore, RetrievedChunk

WINDOW_START = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)


class FakeEmbedder:
    model = "fake-embed"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def _hit(
    chunk_id: str,
    comment_id: int,
    story_id: int,
    text: str,
    distance: float,
    title: str = "Thread",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document=text,
        distance=distance,
        metadata={
            "comment_id": comment_id,
            "story_id": story_id,
            "story_title": title,
            "created_at": WINDOW_START.isoformat(),
            "source_url": f"https://news.ycombinator.com/item?id={comment_id}",
        },
    )


def test_select_evidence_caps_per_thread_and_dedupes() -> None:
    hits = [
        _hit("hn-1-0", 1, 10, "first rust take", 0.1, "Rust"),
        _hit("hn-1-0", 1, 10, "duplicate chunk", 0.11, "Rust"),
        _hit("hn-2-0", 1, 10, "same comment other chunk", 0.12, "Rust"),
        _hit("hn-3-0", 3, 10, "second rust take", 0.13, "Rust"),
        _hit("hn-4-0", 4, 10, "third rust take should drop", 0.14, "Rust"),
        _hit("hn-5-0", 5, 11, "python take", 0.2, "Python"),
    ]
    selected = select_evidence(hits, keep_count=8, per_thread=2)
    assert [item.chunk_id for item in selected] == ["hn-1-0", "hn-3-0", "hn-5-0"]
    assert selected[0].story_title == "Rust"
    assert selected[0].source_url.endswith("id=1")
    assert selected[0].text == "first rust take"
    assert selected[0].comment_id == 1


def test_theme_retrieval_respects_report_window(tmp_path: Path) -> None:
    inside = PreparedChunk(
        chunk_id="hn-1-0",
        comment_id=1,
        story_id=10,
        story_title="Rust",
        created_at=WINDOW_START,
        source_url="https://news.ycombinator.com/item?id=1",
        content_hash=content_hash("Rust borrow checker"),
        chunk_index=0,
        text="Rust borrow checker",
    )
    outside = PreparedChunk(
        chunk_id="hn-2-0",
        comment_id=2,
        story_id=11,
        story_title="Old rust thread",
        created_at=WINDOW_START - timedelta(days=8),
        source_url="https://news.ycombinator.com/item?id=2",
        content_hash=content_hash("ancient rust comment"),
        chunk_index=0,
        text="ancient rust comment",
    )
    start_ts = int(WINDOW_START.timestamp())
    end_ts = start_ts + 3600
    with ChunkStore(str(tmp_path / "chroma")) as store:
        store.upsert_chunks(
            [inside, outside],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "fake-embed",
        )
        hits = retrieve_theme_evidence(
            store,
            FakeEmbedder(),
            query="rust memory safety",
            start_ts=start_ts,
            end_ts=end_ts,
            fetch_count=8,
            keep_count=4,
            per_thread=2,
        )
        listed = store.list_chunks(start_ts=start_ts, end_ts=end_ts)
    assert [item.chunk_id for item in hits] == ["hn-1-0"]
    assert hits[0].story_title == "Rust"
    assert [item.chunk_id for item in listed] == ["hn-1-0"]

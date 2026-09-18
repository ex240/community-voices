from datetime import UTC, datetime

from app.overview import build_corpus_overview
from app.store import RetrievedChunk

WINDOW_START = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 17, 17, 0, tzinfo=UTC)


def _chunk(
    chunk_id: str,
    comment_id: int,
    story_id: int,
    text: str,
    title: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document=text,
        distance=0.0,
        metadata={
            "comment_id": comment_id,
            "story_id": story_id,
            "story_title": title,
            "created_at": WINDOW_START.isoformat(),
            "created_at_i": int(WINDOW_START.timestamp()),
            "source_url": f"https://news.ycombinator.com/item?id={comment_id}",
        },
    )


def test_overview_uses_stable_thread_order_and_measured_counts() -> None:
    long_text = "A" * 400
    chunks = [
        _chunk("hn-20-0", 20, 200, long_text, "Later thread"),
        _chunk("hn-10-0", 10, 100, "Short take on rust", "Earlier thread"),
        _chunk("hn-11-0", 11, 100, "Another rust comment", "Earlier thread"),
    ]
    overview = build_corpus_overview(
        chunks,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        max_threads=2,
        snippet_chars=40,
    )
    assert overview.story_ids == [100, 200]
    assert overview.thread_count == 2
    assert overview.comment_count == 3
    assert overview.chunk_count == 3
    assert "story_id=100" in overview.text
    assert "comments_in_sample=2" in overview.text
    assert long_text not in overview.text
    assert "…" in overview.text
    assert "site-wide popularity" in overview.text


def test_overview_caps_threads_instead_of_listing_every_story() -> None:
    chunks = [
        _chunk(f"hn-{story}-0", story, story, f"comment {story}", f"Thread {story}")
        for story in (30, 10, 20, 40)
    ]
    overview = build_corpus_overview(
        chunks,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        max_threads=2,
        snippet_chars=80,
    )
    assert overview.story_ids == [10, 20]
    assert "story_id=30" not in overview.text
    assert "story_id=40" not in overview.text

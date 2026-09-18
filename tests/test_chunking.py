from datetime import UTC, datetime

from app.chunking import chunk_id_for, chunks_from_comment, content_hash, split_text
from app.hn_client import CommentExcerpt


def test_content_hash_is_stable_and_content_sensitive() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hello ")
    assert len(content_hash("hello")) == 64


def test_split_text_keeps_short_comments_whole() -> None:
    assert split_text("short comment", 100) == ["short comment"]


def test_split_text_breaks_long_comments_on_newlines() -> None:
    text = "first paragraph\n\nsecond paragraph that continues"
    chunks = split_text(text, 20)
    assert chunks[0] == "first paragraph"
    assert "second paragraph" in chunks[1]
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_chunks_from_comment_use_stable_ids_and_hashes() -> None:
    comment = CommentExcerpt(
        comment_id=42,
        story_id=7,
        story_title="Example",
        created_at=datetime(2026, 9, 10, tzinfo=UTC),
        author="alice",
        url="https://news.ycombinator.com/item?id=42",
        text="alpha " * 40,
    )
    chunks = chunks_from_comment(comment, max_chars=30)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == chunk_id_for(42, 0)
    assert chunks[1].chunk_id == chunk_id_for(42, 1)
    assert chunks[0].content_hash == content_hash(chunks[0].text)
    assert chunks[0].comment_id == 42
    assert chunks[0].story_id == 7
    assert chunks[0].source_url.endswith("id=42")

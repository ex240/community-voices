from pathlib import Path

from app.chunking import PreparedChunk, content_hash
from app.store import ChunkStore
from tests.test_ingest import WINDOW_START


def test_retrieval_returns_nearest_synthetic_vector(tmp_path: Path) -> None:
    cats = PreparedChunk(
        chunk_id="hn-1-0",
        comment_id=1,
        story_id=10,
        story_title="Cats",
        created_at=WINDOW_START,
        source_url="https://news.ycombinator.com/item?id=1",
        content_hash=content_hash("I like cats"),
        chunk_index=0,
        text="I like cats",
    )
    planes = PreparedChunk(
        chunk_id="hn-2-0",
        comment_id=2,
        story_id=11,
        story_title="Airplanes",
        created_at=WINDOW_START,
        source_url="https://news.ycombinator.com/item?id=2",
        content_hash=content_hash("Jet engines are loud"),
        chunk_index=0,
        text="Jet engines are loud",
    )
    with ChunkStore(str(tmp_path / "chroma")) as store:
        store.upsert_chunks([cats, planes], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "synthetic-test")
        hits = store.query([0.95, 0.05, 0.0], n_results=1)
        assert hits[0].chunk_id == "hn-1-0"
        assert hits[0].metadata["comment_id"] == 1
        assert hits[0].metadata["story_title"] == "Cats"
        assert hits[0].metadata["source_url"].endswith("id=1")
        assert "cats" in hits[0].document
        later_hits = store.query(
            [0.95, 0.05, 0.0],
            n_results=1,
            start_ts=int(WINDOW_START.timestamp()) + 10,
            end_ts=int(WINDOW_START.timestamp()) + 20,
        )
        assert later_hits == []

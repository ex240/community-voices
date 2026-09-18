"""Retrieve a few stored HN chunks with the same embedding model used at ingest."""

from __future__ import annotations

import sys

from app.config import get_settings
from app.embedder import OpenAIEmbedder
from app.hn_client import shorten
from app.store import ChunkStore

QUERY = "What topics are Hacker News commenters discussing this week?"
RESULT_COUNT = 5


def main() -> int:
    settings = get_settings()
    if not settings.api_key_configured:
        print(
            "Retrieval smoke needs OPENAI_API_KEY so the query uses the same embedding model. "
            "Copy .env.example to .env and add a key.",
            file=sys.stderr,
        )
        return 1

    with ChunkStore(settings.chroma_path) as store:
        count = store.count()
        if count == 0:
            print("No chunks in Chroma. Run python -m app.ingest first.", file=sys.stderr)
            return 1

        embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
        query_vector = embedder.embed_texts([QUERY])[0]
        hits = store.query(query_vector, n_results=min(RESULT_COUNT, count))

    print("Community Voices retrieval smoke")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Query: {QUERY}")
    print(f"Chunks in store: {count}")
    print(f"Results: {len(hits)}")
    print("Distance is Chroma cosine distance (0 is nearest). It is not popularity or confidence.")
    print()
    for index, hit in enumerate(hits, start=1):
        meta = hit.metadata
        print(f"{index}. {hit.chunk_id}  distance={hit.distance:.4f}")
        print(f"   story_id={meta.get('story_id')}  comment_id={meta.get('comment_id')}")
        print(f"   title: {meta.get('story_title')}")
        print(f"   created_at: {meta.get('created_at')}")
        print(f"   url: {meta.get('source_url')}")
        print(f"   hash: {meta.get('content_hash')}")
        print(f"   {shorten(hit.document)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

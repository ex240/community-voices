"""Local Chroma persistence for comment chunks. Callers pass embeddings in."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunking import PreparedChunk

COLLECTION_NAME = "hn_comment_chunks"


def corpus_snapshot(path: str) -> dict[str, object]:
    with ChunkStore(path) as store:
        count = store.count()
    return {
        "chunk_count": count,
        "status": "ready" if count else "empty",
    }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document: str
    distance: float
    metadata: dict[str, Any]


class ChunkStore:
    def __init__(self, path: str) -> None:
        self.path = str(Path(path))
        self._client = chromadb.PersistentClient(
            path=self.path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())

    def get_metadata(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        result = self._collection.get(ids=ids, include=["metadatas"])
        found: dict[str, dict[str, Any]] = {}
        for chunk_id, metadata in zip(result.get("ids") or [], result.get("metadatas") or []):
            if chunk_id is not None and metadata is not None:
                found[str(chunk_id)] = metadata
        return found

    def list_ids(self) -> list[str]:
        ids: list[str] = []
        offset = 0
        page_size = 1000
        while True:
            result = self._collection.get(include=["metadatas"], limit=page_size, offset=offset)
            batch = result.get("ids") or []
            ids.extend(str(item) for item in batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return ids

    def list_chunks(
        self,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return stored chunks, optionally filtered to [start_ts, end_ts)."""
        where = _window_where(start_ts, end_ts)
        chunks: list[RetrievedChunk] = []
        offset = 0
        page_size = 1000
        while True:
            kwargs: dict[str, Any] = {
                "include": ["documents", "metadatas"],
                "limit": page_size,
                "offset": offset,
            }
            if where is not None:
                kwargs["where"] = where
            result = self._collection.get(**kwargs)
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for chunk_id, document, metadata in zip(ids, documents, metadatas):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=str(chunk_id),
                        document=document or "",
                        distance=0.0,
                        metadata=metadata or {},
                    )
                )
            if len(ids) < page_size:
                break
            offset += page_size
        return chunks

    def ids_for_comment(self, comment_id: int) -> list[str]:
        result = self._collection.get(
            where={"comment_id": comment_id},
            include=["metadatas"],
        )
        return [str(item) for item in (result.get("ids") or [])]

    def upsert_chunks(
        self,
        chunks: list[PreparedChunk],
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_metadata_for(chunk, embedding_model) for chunk in chunks],
        )

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def query(
        self,
        query_embedding: list[float],
        *,
        n_results: int,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[RetrievedChunk]:
        where = _window_where(start_ts, end_ts)
        count = self.count()
        if count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        retrieved: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    document=document or "",
                    distance=float(distance),
                    metadata=metadata or {},
                )
            )
        return retrieved

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ChunkStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _metadata_for(chunk: PreparedChunk, embedding_model: str) -> dict[str, Any]:
    return {
        "comment_id": chunk.comment_id,
        "story_id": chunk.story_id,
        "story_title": chunk.story_title,
        "created_at": chunk.created_at.isoformat(),
        "created_at_i": int(chunk.created_at.timestamp()),
        "source_url": chunk.source_url,
        "content_hash": chunk.content_hash,
        "chunk_index": chunk.chunk_index,
        "embedding_model": embedding_model,
    }


def _window_where(start_ts: int | None, end_ts: int | None) -> dict[str, Any] | None:
    if start_ts is None and end_ts is None:
        return None
    clauses: list[dict[str, Any]] = []
    if start_ts is not None:
        clauses.append({"created_at_i": {"$gte": start_ts}})
    if end_ts is not None:
        clauses.append({"created_at_i": {"$lt": end_ts}})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

"""Embedding helpers. The OpenAI client is created only when this class is constructed."""

from __future__ import annotations

from typing import Protocol

EMBED_BATCH_SIZE = 64


class Embedder(Protocol):
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input string, in the same order."""


class OpenAIEmbedder:
    """Thin wrapper around the official OpenAI embeddings API."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is not configured")
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [item.replace("\n", " ") for item in texts[start : start + EMBED_BATCH_SIZE]]
            response = self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda row: row.index)
            vectors.extend(item.embedding for item in ordered)
        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding provider returned a different number of vectors than inputs."
            )
        return vectors

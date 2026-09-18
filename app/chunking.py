"""Turn cleaned comment text into stable, retrieval-friendly chunks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from app.hn_client import CommentExcerpt


def content_hash(text: str) -> str:
    """SHA-256 of the exact text we store. Same bytes → same hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id_for(comment_id: int, index: int) -> str:
    return f"hn-{comment_id}-{index}"


def split_text(text: str, max_chars: int) -> list[str]:
    """Split long comments on newlines/spaces; keep short comments as one chunk."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]

    chunks: list[str] = []
    remaining = stripped
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        window = remaining[:max_chars]
        cut = window.rfind("\n")
        if cut < max_chars // 4:
            cut = window.rfind(" ")
        if cut < max_chars // 4:
            cut = max_chars
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].lstrip()
    return chunks


@dataclass(frozen=True)
class PreparedChunk:
    chunk_id: str
    comment_id: int
    story_id: int
    story_title: str
    created_at: datetime
    source_url: str
    content_hash: str
    chunk_index: int
    text: str


def chunks_from_comment(comment: CommentExcerpt, max_chars: int) -> list[PreparedChunk]:
    pieces = split_text(comment.text, max_chars)
    prepared: list[PreparedChunk] = []
    for index, piece in enumerate(pieces):
        prepared.append(
            PreparedChunk(
                chunk_id=chunk_id_for(comment.comment_id, index),
                comment_id=comment.comment_id,
                story_id=comment.story_id,
                story_title=comment.story_title,
                created_at=comment.created_at,
                source_url=comment.url,
                content_hash=content_hash(piece),
                chunk_index=index,
                text=piece,
            )
        )
    return prepared

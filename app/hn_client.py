"""Read-only Hacker News access via the Algolia Search API.

This module does not embed text, call a writing model, or write to Chroma.
Comment timestamps are checked against a half-open UTC window [start, end).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any

import httpx

ALGOLIA_BASE_URL = "https://hn.algolia.com/api/v1"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={item_id}"
DEFAULT_USER_AGENT = "CommunityVoices/0.1 (local source-smoke; educational)"
DEFAULT_MAX_REQUESTS = 6
EXCERPT_LIMIT = 240


class HackerNewsError(Exception):
    """Raised when HN source access fails or the request budget is exhausted."""


class RequestBudgetError(HackerNewsError):
    """Raised when the bounded HN request budget is used up."""


def utc_lookback_window(now: datetime, lookback_days: int) -> tuple[datetime, datetime]:
    """Return a timezone-aware UTC interval [start, end) covering lookback_days."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    end = now.astimezone(UTC)
    start = end - timedelta(days=lookback_days)
    return start, end


def is_in_window(moment: datetime, start: datetime, end: datetime) -> bool:
    """True when start <= moment < end. All three values must be timezone-aware."""
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    moment_utc = moment.astimezone(UTC)
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    return start_utc <= moment_utc < end_utc


def html_to_text(raw: str | None) -> str:
    """Convert source HTML to readable text without executing it."""
    if not raw:
        return ""
    parser = _HtmlToText()
    parser.feed(raw)
    parser.close()
    lines = [" ".join(line.split()) for line in parser.text().splitlines()]
    return "\n".join(line for line in lines if line).strip()


def shorten(text: str, limit: int = EXCERPT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def hn_item_url(item_id: int) -> str:
    return HN_ITEM_URL.format(item_id=item_id)


@dataclass(frozen=True)
class CommentExcerpt:
    comment_id: int
    story_id: int
    story_title: str
    created_at: datetime
    author: str | None
    url: str
    text: str

    @property
    def excerpt(self) -> str:
        return shorten(self.text)


@dataclass(frozen=True)
class ClassifiedHit:
    status: str
    comment_id: int | None
    comment: CommentExcerpt | None


@dataclass(frozen=True)
class DiscussionSample:
    story_id: int
    story_title: str
    story_url: str
    comments: list[CommentExcerpt]
    requests_used: int


class HackerNewsClient:
    """Bounded Algolia client. Callers inject httpx so tests can mock HTTP."""

    def __init__(
        self,
        http: httpx.Client,
        *,
        max_requests: int = DEFAULT_MAX_REQUESTS,
    ) -> None:
        self._http = http
        self._max_requests = max_requests
        self.requests_used = 0

    def find_discussion_sample(
        self,
        start: datetime,
        end: datetime,
        *,
        max_comments: int,
    ) -> DiscussionSample:
        """Find one in-window discussion and a small set of comment excerpts."""
        if max_comments <= 0:
            raise ValueError("max_comments must be positive")

        discovery_hits = self.search_comments(start, end, hits_per_page=20)
        tried_story_ids: set[int] = set()

        for hit in discovery_hits:
            story_id = _optional_int(hit.get("story_id"))
            if story_id is None or story_id in tried_story_ids:
                continue
            tried_story_ids.add(story_id)

            story_hits = self.search_comments(
                start,
                end,
                hits_per_page=max_comments,
                story_id=story_id,
            )
            comments = parse_comment_hits(story_hits, start, end)
            if not comments:
                continue

            title = comments[0].story_title or str(hit.get("story_title") or "Untitled")
            return DiscussionSample(
                story_id=story_id,
                story_title=title,
                story_url=hn_item_url(story_id),
                comments=comments,
                requests_used=self.requests_used,
            )

        raise HackerNewsError("No Hacker News discussion with usable in-window comments was found.")

    def search_comments(
        self,
        start: datetime,
        end: datetime,
        *,
        hits_per_page: int,
        story_id: int | None = None,
    ) -> list[dict[str, Any]]:
        start_utc, end_utc = _require_window(start, end)
        tags = f"comment,story_{story_id}" if story_id is not None else "comment"
        payload = self._get_json(
            "/search_by_date",
            {
                "tags": tags,
                "numericFilters": (
                    f"created_at_i>={int(start_utc.timestamp())},"
                    f"created_at_i<{int(end_utc.timestamp())}"
                ),
                "hitsPerPage": hits_per_page,
            },
        )
        hits = payload.get("hits")
        if hits is None:
            return []
        if not isinstance(hits, list):
            raise HackerNewsError("Unexpected Algolia response: hits is not a list.")
        return hits

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.requests_used >= self._max_requests:
            raise RequestBudgetError(
                f"Request budget exhausted after {self._max_requests} HN requests."
            )
        self.requests_used += 1
        url = f"{ALGOLIA_BASE_URL}{path}"
        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise HackerNewsError(f"Hacker News request failed: {exc}") from exc
        except ValueError as exc:
            raise HackerNewsError(f"Hacker News response was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise HackerNewsError("Unexpected Algolia response: expected a JSON object.")
        return payload


def parse_comment_hits(
    hits: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> list[CommentExcerpt]:
    """Keep comments with readable text whose timestamps fall in [start, end)."""
    excerpts: list[CommentExcerpt] = []
    for hit in hits:
        classified = classify_comment_hit(hit, start, end)
        if classified.status == "ok" and classified.comment is not None:
            excerpts.append(classified.comment)
    return excerpts


def classify_comment_hit(
    hit: dict[str, Any],
    start: datetime,
    end: datetime,
) -> ClassifiedHit:
    """Label a raw Algolia hit so ingest can count skips and deletions."""
    comment_id = _optional_int(hit.get("objectID") or hit.get("id"))
    raw_text = hit.get("comment_text")
    created_at = _parse_created_at(hit)
    if created_at is None:
        return ClassifiedHit("invalid", comment_id, None)
    if not is_in_window(created_at, start, end):
        return ClassifiedHit("out_of_window", comment_id, None)

    story_id = _optional_int(hit.get("story_id"))
    if comment_id is None or story_id is None:
        return ClassifiedHit("invalid", comment_id, None)

    if raw_text is None:
        return ClassifiedHit("deleted", comment_id, None)

    text = html_to_text(raw_text)
    if text in {"[deleted]", "[dead]"}:
        return ClassifiedHit("deleted", comment_id, None)
    if not text:
        return ClassifiedHit("empty", comment_id, None)

    title = str(hit.get("story_title") or "Untitled")
    author = hit.get("author")
    comment = CommentExcerpt(
        comment_id=comment_id,
        story_id=story_id,
        story_title=title,
        created_at=created_at,
        author=str(author) if author else None,
        url=hn_item_url(comment_id),
        text=text,
    )
    return ClassifiedHit("ok", comment_id, comment)


def split_window(start: datetime, end: datetime, parts: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into contiguous half-open buckets."""
    start_utc, end_utc = _require_window(start, end)
    if parts <= 0:
        raise ValueError("parts must be positive")
    span = (end_utc - start_utc) / parts
    buckets: list[tuple[datetime, datetime]] = []
    cursor = start_utc
    for index in range(parts):
        bucket_end = end_utc if index == parts - 1 else start_utc + span * (index + 1)
        buckets.append((cursor, bucket_end))
        cursor = bucket_end
    return buckets


def story_ids_from_hits(hits: list[dict[str, Any]]) -> list[int]:
    """Preserve first-seen story order from a page of comment hits."""
    story_ids: list[int] = []
    seen: set[int] = set()
    for hit in hits:
        story_id = _optional_int(hit.get("story_id"))
        if story_id is None or story_id in seen:
            continue
        seen.add(story_id)
        story_ids.append(story_id)
    return story_ids


def round_robin_ids(groups: list[list[int]], limit: int) -> list[int]:
    """Take IDs from each group in turn so one day cannot dominate the sample."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    queues = [list(group) for group in groups if group]
    selected: list[int] = []
    seen: set[int] = set()
    while queues and len(selected) < limit:
        group = queues.pop(0)
        story_id = group.pop(0)
        if story_id not in seen:
            seen.add(story_id)
            selected.append(story_id)
        if group:
            queues.append(group)
    return selected


def select_hits_by_id(hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the lowest objectIDs so the bounded sample does not chase newest comments."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = sorted(
        hits,
        key=lambda hit: _optional_int(hit.get("objectID") or hit.get("id")) or 0,
    )
    return ranked[:limit]


def parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO timestamp. Naive values are treated as UTC."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def resolve_ingest_window(
    *,
    lookback_days: int,
    now: datetime,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Build [start, end) from optional CLI bounds. Default is a rolling lookback from now."""
    if start is None and end is None:
        return utc_lookback_window(now, lookback_days)
    if end is None:
        if start is None:
            raise ValueError("start or end is required")
        start_utc = start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC)
        return _require_window(start_utc, start_utc + timedelta(days=lookback_days))
    if start is None:
        end_utc = end.astimezone(UTC) if end.tzinfo else end.replace(tzinfo=UTC)
        return _require_window(end_utc - timedelta(days=lookback_days), end_utc)
    start_utc = start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC)
    end_utc = end.astimezone(UTC) if end.tzinfo else end.replace(tzinfo=UTC)
    return _require_window(start_utc, end_utc)


def _parse_created_at(hit: dict[str, Any]) -> datetime | None:
    timestamp = hit.get("created_at_i")
    if timestamp is not None:
        try:
            return datetime.fromtimestamp(int(timestamp), tz=UTC)
        except (TypeError, ValueError, OverflowError, OSError):
            return None
    raw = hit.get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    if not start_utc < end_utc:
        raise ValueError("start must be earlier than end")
    return start_utc, end_utc


class _HtmlToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif tag in {"p", "br", "div", "li", "tr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "div", "li"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)

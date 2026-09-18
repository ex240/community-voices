"""CLI smoke check: can we read a bounded sample of recent HN comments?"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import httpx

from app.config import get_settings
from app.hn_client import (
    DEFAULT_MAX_REQUESTS,
    DEFAULT_USER_AGENT,
    HackerNewsClient,
    HackerNewsError,
    utc_lookback_window,
)


def main() -> int:
    settings = get_settings()
    start, end = utc_lookback_window(
        datetime.now(UTC),
        settings.lookback_days,
    )
    print("Community Voices source smoke check")
    print(f"UTC window [start, end): {start.isoformat()} -> {end.isoformat()}")
    print(f"Request budget: {DEFAULT_MAX_REQUESTS}")

    timeout = httpx.Timeout(settings.http_timeout_seconds)
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    max_comments = min(5, settings.max_comments_per_thread)

    try:
        with httpx.Client(timeout=timeout, headers=headers) as http:
            client = HackerNewsClient(http, max_requests=DEFAULT_MAX_REQUESTS)
            sample = client.find_discussion_sample(
                start,
                end,
                max_comments=max_comments,
            )
    except HackerNewsError as exc:
        print(f"Source smoke check failed: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"Source smoke check failed: {exc}", file=sys.stderr)
        return 1

    print(f"Requests used: {sample.requests_used}/{DEFAULT_MAX_REQUESTS}")
    print()
    print("Story")
    print(f"  title: {sample.story_title}")
    print(f"  id: {sample.story_id}")
    print(f"  url: {sample.story_url}")
    print()
    print(f"Comments ({len(sample.comments)})")
    for comment in sample.comments:
        print(f"  [{comment.comment_id}] {comment.created_at.isoformat()}  {comment.url}")
        if comment.author:
            print(f"    author: {comment.author}")
        print(f"    {comment.excerpt}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

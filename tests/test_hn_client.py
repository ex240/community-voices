from datetime import UTC, datetime

import httpx
import pytest

from app.hn_client import (
    HackerNewsClient,
    HackerNewsError,
    classify_comment_hit,
    html_to_text,
    parse_comment_hits,
    parse_utc_datetime,
    resolve_ingest_window,
    round_robin_ids,
    select_hits_by_id,
    story_ids_from_hits,
)

WINDOW_START = datetime(2026, 9, 9, 18, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 16, 18, 0, tzinfo=UTC)


def test_html_to_text_strips_tags_and_skips_script() -> None:
    raw = "<p>Hello &amp; welcome</p><script>alert('nope')</script><p>Second line<br>continues</p>"
    text = html_to_text(raw)
    assert "Hello & welcome" in text
    assert "alert" not in text
    assert "Second line" in text
    assert "continues" in text


def test_html_to_text_handles_empty_and_deleted() -> None:
    assert html_to_text(None) == ""
    assert html_to_text("") == ""
    assert html_to_text("[deleted]") == "[deleted]"


def test_parse_comment_hits_skips_outside_window_and_empty_text() -> None:
    hits = [
        {
            "objectID": "10",
            "story_id": 1,
            "story_title": "In window",
            "author": "alice",
            "created_at_i": int(WINDOW_START.timestamp()),
            "comment_text": "<p>Useful comment</p>",
        },
        {
            "objectID": "11",
            "story_id": 1,
            "story_title": "At end",
            "created_at_i": int(WINDOW_END.timestamp()),
            "comment_text": "<p>Too late</p>",
        },
        {
            "objectID": "12",
            "story_id": 1,
            "story_title": "Empty",
            "created_at_i": int(WINDOW_START.timestamp()) + 60,
            "comment_text": "<p>   </p>",
        },
        {
            "objectID": "13",
            "story_id": 1,
            "story_title": "Deleted",
            "created_at_i": int(WINDOW_START.timestamp()) + 120,
            "comment_text": None,
        },
    ]
    excerpts = parse_comment_hits(hits, WINDOW_START, WINDOW_END)
    assert len(excerpts) == 1
    assert excerpts[0].comment_id == 10
    assert excerpts[0].excerpt == "Useful comment"
    assert excerpts[0].url.endswith("id=10")


def test_classify_comment_hit_labels_deleted_and_empty() -> None:
    deleted = classify_comment_hit(
        {
            "objectID": "13",
            "story_id": 1,
            "created_at_i": int(WINDOW_START.timestamp()) + 120,
            "comment_text": None,
        },
        WINDOW_START,
        WINDOW_END,
    )
    empty = classify_comment_hit(
        {
            "objectID": "12",
            "story_id": 1,
            "created_at_i": int(WINDOW_START.timestamp()) + 60,
            "comment_text": "<p>   </p>",
        },
        WINDOW_START,
        WINDOW_END,
    )
    assert deleted.status == "deleted"
    assert empty.status == "empty"


def test_round_robin_avoids_one_group_dominating() -> None:
    assert round_robin_ids([[1, 2, 3], [10, 11], [20]], 4) == [1, 10, 20, 2]


def test_story_ids_from_hits_preserve_first_seen_order() -> None:
    hits = [
        {"story_id": 5},
        {"story_id": "5"},
        {"story_id": 9},
        {"story_id": None},
    ]
    assert story_ids_from_hits(hits) == [5, 9]


def test_select_hits_by_id_keeps_lowest_object_ids() -> None:
    hits = [{"objectID": "30"}, {"objectID": "10"}, {"objectID": "20"}]
    selected = select_hits_by_id(hits, 2)
    assert [hit["objectID"] for hit in selected] == ["10", "20"]


def test_resolve_ingest_window_can_pin_end() -> None:
    now = datetime(2026, 9, 17, 18, 0, tzinfo=UTC)
    end = datetime(2026, 9, 17, 17, 0, tzinfo=UTC)
    start, resolved_end = resolve_ingest_window(lookback_days=7, now=now, end=end)
    assert resolved_end == end
    assert start == datetime(2026, 9, 10, 17, 0, tzinfo=UTC)


def test_resolve_ingest_window_treats_naive_start_and_end_as_utc() -> None:
    start, end = resolve_ingest_window(
        lookback_days=7,
        now=datetime(2026, 9, 17, 18, 0, tzinfo=UTC),
        start=datetime(2026, 9, 10, 17, 0),
        end=datetime(2026, 9, 17, 17, 0),
    )
    assert start == datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
    assert end == datetime(2026, 9, 17, 17, 0, tzinfo=UTC)


def test_parse_utc_datetime_accepts_z_suffix() -> None:
    parsed = parse_utc_datetime("2026-09-17T17:00:00Z")
    assert parsed == datetime(2026, 9, 17, 17, 0, tzinfo=UTC)


def test_request_failure_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = HackerNewsClient(http, max_requests=2)
        with pytest.raises(HackerNewsError, match="Hacker News request failed"):
            client.find_discussion_sample(WINDOW_START, WINDOW_END, max_comments=3)


def test_find_discussion_sample_uses_story_filter() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        tags = request.url.params.get("tags")
        if tags == "comment":
            return httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "objectID": "21",
                            "story_id": 99,
                            "story_title": "Recent thread",
                            "created_at_i": int(WINDOW_START.timestamp()) + 10,
                            "comment_text": "<p>Discovery hit</p>",
                        }
                    ]
                },
            )
        if tags == "comment,story_99":
            return httpx.Response(
                200,
                json={
                    "hits": [
                        {
                            "objectID": "22",
                            "story_id": 99,
                            "story_title": "Recent thread",
                            "author": "bob",
                            "created_at_i": int(WINDOW_START.timestamp()) + 20,
                            "comment_text": "<p>Grounded excerpt</p>",
                        }
                    ]
                },
            )
        return httpx.Response(400, json={"message": "unexpected tags"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = HackerNewsClient(http, max_requests=6)
        sample = client.find_discussion_sample(
            WINDOW_START,
            WINDOW_END,
            max_comments=3,
        )

    assert sample.story_id == 99
    assert sample.story_title == "Recent thread"
    assert sample.comments[0].excerpt == "Grounded excerpt"
    assert sample.requests_used == 2
    assert any("story_99" in url for url in calls)

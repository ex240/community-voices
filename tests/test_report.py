import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.chunking import PreparedChunk, content_hash
from app.config import Settings
from app.evidence import EvidenceItem
from app.generate_report import generate_rag_report, resolve_report_window, run_generate_report
from app.report import (
    REPORT_INSTRUCTIONS,
    CitationValidationError,
    ThemeBundle,
    UnmeasuredStrengthError,
    apply_citation_guard,
    assign_citation_ids,
    load_report,
    normalize_citations,
    parse_discovered_themes,
    parse_written_report,
    reject_unmeasured_strength,
    save_report,
    unmeasured_strength_phrases,
)
from app.store import ChunkStore

WINDOW_START = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 17, 17, 0, tzinfo=UTC)


class FakeEmbedder:
    model = "fake-embed"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class ScriptedWriter:
    model = "fake-writer"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, instructions: str, input_text: str) -> str:
        self.calls.append((instructions, input_text))
        if not self.replies:
            raise RuntimeError("writing model failed")
        return self.replies.pop(0)


def _chunk(
    comment_id: int,
    story_id: int,
    text: str,
    title: str,
    created_at: datetime = WINDOW_START,
) -> PreparedChunk:
    return PreparedChunk(
        chunk_id=f"hn-{comment_id}-0",
        comment_id=comment_id,
        story_id=story_id,
        story_title=title,
        created_at=created_at,
        source_url=f"https://news.ycombinator.com/item?id={comment_id}",
        content_hash=content_hash(text),
        chunk_index=0,
        text=text,
    )


def _evidence(chunk_id: str, comment_id: int, story_id: int, text: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        comment_id=comment_id,
        story_id=story_id,
        story_title="Rust",
        source_url=f"https://news.ycombinator.com/item?id={comment_id}",
        created_at=WINDOW_START.isoformat(),
        text=text,
        distance=0.2,
    )


def test_theme_discovery_parses_and_caps_themes() -> None:
    raw = json.dumps(
        {
            "themes": [
                {
                    "title": "Rust",
                    "description": "Borrow checker",
                    "retrieval_query": "rust memory",
                },
                {"title": "Skip me"},
                {"title": "AI", "description": "Models", "retrieval_query": "language models"},
                {"title": "Extra", "description": "Too many", "retrieval_query": "extra"},
            ]
        }
    )
    themes = parse_discovered_themes(raw, max_themes=2)
    assert [theme.title for theme in themes] == ["Rust", "AI"]
    assert themes[0].retrieval_query == "rust memory"


def test_theme_discovery_rejects_malformed_output() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_discovered_themes("{", max_themes=6)
    with pytest.raises(ValueError, match="themes array"):
        parse_discovered_themes('{"themes": "nope"}', max_themes=6)


def test_written_report_parse_skips_incomplete_rows() -> None:
    retrospective, predictions = parse_written_report(
        json.dumps(
            {
                "retrospective": [
                    {"title": "Rust", "body": "People discussed lifetimes [S1]."},
                    {"title": "Missing body"},
                ],
                "predictions": [
                    {
                        "claim": "Lifetimes may stay in the conversation.",
                        "rationale": "The sample already returns to it [S1].",
                        "uncertainty": "This is a prediction.",
                    },
                    {"rationale": "no claim"},
                ],
            }
        )
    )
    assert len(retrospective) == 1
    assert len(predictions) == 1
    assert "[S1]" in retrospective[0].body


def test_unknown_citations_reject_the_report() -> None:
    bundles = [
        ThemeBundle(
            title="Rust",
            description="memory",
            retrieval_query="rust",
            evidence=[_evidence("hn-1-0", 1, 10, "lifetimes matter")],
        )
    ]
    sources = assign_citation_ids(bundles)
    retrospective, predictions = parse_written_report(
        json.dumps(
            {
                "retrospective": [
                    {"title": "Rust", "body": "Lifetimes came up [S1] and also [S99]."}
                ],
                "predictions": [
                    {
                        "claim": "It may continue [S1].",
                        "rationale": "Invented [S2].",
                        "uncertainty": "Uncertain.",
                    }
                ],
            }
        )
    )
    with pytest.raises(CitationValidationError, match="S99") as caught:
        apply_citation_guard(retrospective, predictions, set(sources))
    assert caught.value.unknown_ids == ["S99", "S2"]
    assert "[S99]" in retrospective[0].body


def test_grouped_citations_are_split_when_all_ids_are_valid() -> None:
    from app.report import Prediction, ThemeSection

    text, unknown = normalize_citations("Hiring mentions Python [S1, S2, S4].", {"S1", "S2", "S4"})
    assert text == "Hiring mentions Python [S1] [S2] [S4]."
    assert unknown == []
    sections, predictions = apply_citation_guard(
        [ThemeSection(title="Hiring", body="Sampled posts mention Python [S1, S2].")],
        [
            Prediction(
                claim="Hiring posts may continue [S1].",
                rationale="Several sampled comments list it [S2].",
                uncertainty="Uncertain.",
            )
        ],
        {"S1", "S2"},
    )
    assert sections[0].body == "Sampled posts mention Python [S1] [S2]."
    assert predictions[0].rationale == "Several sampled comments list it [S2]."


def test_grouped_citations_with_unknown_ids_are_not_silently_rewritten() -> None:
    from app.report import ThemeSection

    original = "Hiring mentions Python [S1, S2, S99]."
    text, unknown = normalize_citations(original, {"S1", "S2"})
    assert text == original
    assert unknown == ["S99"]
    with pytest.raises(CitationValidationError, match="S99"):
        apply_citation_guard(
            [ThemeSection(title="Hiring", body=original)],
            [],
            {"S1", "S2"},
        )


def test_report_persistence_round_trip(tmp_path: Path) -> None:
    from app.report import GeneratedReport, Prediction, SourceRecord, ThemeSection

    path = tmp_path / "latest_report.json"
    report = GeneratedReport(
        kind="rag",
        window_start=WINDOW_START.isoformat(),
        window_end=WINDOW_END.isoformat(),
        generated_at=WINDOW_END.isoformat(),
        model="fake-writer",
        embedding_model="fake-embed",
        chunk_count=1,
        thread_count=1,
        comment_count=1,
        overview_story_ids=[10],
        themes=[{"title": "Rust"}],
        retrospective=[ThemeSection(title="Rust", body="Lifetimes [S1].")],
        predictions=[Prediction(claim="May continue.", rationale="[S1]", uncertainty="Uncertain.")],
        limitation="Bounded sample.",
        sources=[
            SourceRecord(
                citation_id="S1",
                chunk_id="hn-1-0",
                comment_id=1,
                story_id=10,
                story_title="Rust",
                source_url="https://news.ycombinator.com/item?id=1",
                created_at=WINDOW_START.isoformat(),
                text="lifetimes matter",
                distance=0.1,
            )
        ],
        unknown_citation_ids=[],
    )
    save_report(str(path), report)
    loaded = load_report(str(path))
    assert loaded is not None
    assert loaded["kind"] == "rag"
    assert loaded["sources"][0]["chunk_id"] == "hn-1-0"
    assert load_report(str(tmp_path / "missing.json")) is None


def test_resolve_window_uses_ingest_manifest(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    manifest = tmp_path / "last_ingest_sample.json"
    manifest.write_text(
        json.dumps(
            {
                "window_start": WINDOW_START.isoformat(),
                "window_end": WINDOW_END.isoformat(),
                "story_ids": [10],
                "comment_ids": [1],
                "chunk_ids": ["hn-1-0"],
            }
        ),
        encoding="utf-8",
    )
    start, end = resolve_report_window(
        chroma_path=str(chroma),
        chunks=[],
        start=None,
        end=None,
    )
    assert start == WINDOW_START
    assert end == WINDOW_END


def test_empty_corpus_raises(tmp_path: Path) -> None:
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        openai_api_key="",
        _env_file=None,
    )
    with pytest.raises(RuntimeError, match="No chunks"):
        run_generate_report(
            settings=settings,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            embedder=FakeEmbedder(),
            writer=ScriptedWriter(["{}"]),
        )


def test_empty_corpus_without_window_raises(tmp_path: Path) -> None:
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        openai_api_key="",
        _env_file=None,
    )
    with pytest.raises(RuntimeError, match="empty"):
        run_generate_report(
            settings=settings,
            embedder=FakeEmbedder(),
            writer=ScriptedWriter(["{}"]),
        )


def test_missing_api_key_raises(tmp_path: Path) -> None:
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        openai_api_key="",
        _env_file=None,
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_generate_report(settings=settings)


def test_model_failure_is_surfaced(tmp_path: Path) -> None:
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        openai_api_key="",
        _env_file=None,
    )
    with ChunkStore(settings.chroma_path) as store:
        store.upsert_chunks(
            [_chunk(1, 10, "Rust lifetimes came up again", "Rust")],
            [[1.0, 0.0, 0.0]],
            "fake-embed",
        )
    with pytest.raises(RuntimeError, match="writing model failed"):
        run_generate_report(
            settings=settings,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            embedder=FakeEmbedder(),
            writer=ScriptedWriter([]),
        )


def test_generate_report_filters_window_and_maps_citations(tmp_path: Path) -> None:
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        openai_api_key="",
        retrieval_per_theme=4,
        evidence_per_thread=2,
        theme_count=4,
        _env_file=None,
    )
    writer = ScriptedWriter(
        [
            json.dumps(
                {
                    "themes": [
                        {
                            "title": "Rust",
                            "description": "Memory safety",
                            "retrieval_query": "rust lifetimes",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "retrospective": [
                        {"title": "Rust", "body": "Commenters returned to lifetimes [S1]."}
                    ],
                    "predictions": [
                        {
                            "claim": "Rust memory safety may stay in view.",
                            "rationale": "The sampled comments already focus on it [S1].",
                            "uncertainty": "This is a prediction, not a measured outcome.",
                        }
                    ],
                }
            ),
        ]
    )
    with ChunkStore(settings.chroma_path) as store:
        store.upsert_chunks(
            [
                _chunk(1, 10, "Rust lifetimes are still painful", "Rust"),
                _chunk(
                    2,
                    11,
                    "Ancient rust comment",
                    "Old Rust",
                    created_at=WINDOW_START - timedelta(days=10),
                ),
            ],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "fake-embed",
        )
        report = generate_rag_report(
            store=store,
            embedder=FakeEmbedder(),
            writer=writer,
            settings=settings,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            now=WINDOW_END,
        )
    assert report.chunk_count == 1
    assert report.overview_story_ids == [10]
    assert report.sources[0].comment_id == 1
    assert report.sources[0].source_url.endswith("id=1")
    assert "[S1]" in report.retrospective[0].body
    assert report.predictions[0].claim.startswith("Rust")
    assert "prediction" in report.predictions[0].uncertainty.lower()
    theme_input = writer.calls[0][1]
    report_input = writer.calls[1][1]
    assert "UNTRUSTED HACKER NEWS OVERVIEW" in theme_input
    assert "json" in theme_input.lower()
    assert "Ancient rust comment" not in theme_input
    assert "UNTRUSTED HACKER NEWS SOURCE TEXT" in report_input
    assert "json" in report_input.lower()
    assert "Rust lifetimes are still painful" in report_input
    assert report.latency_seconds is not None
    assert report.total_tokens is None
    assert "thread title (topic of the sampled thread):" in report_input
    assert "retrieved comment text:" in report_input
    assert "several sampled posts/comments" in writer.calls[1][0]


def test_invalid_citations_are_not_persisted(tmp_path: Path) -> None:
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        report_path=str(tmp_path / "latest_report.json"),
        openai_api_key="",
        retrieval_per_theme=4,
        evidence_per_thread=2,
        theme_count=4,
        _env_file=None,
    )
    report_file = tmp_path / "latest_report.json"
    report_file.write_text('{"kind": "prior"}\n', encoding="utf-8")
    writer = ScriptedWriter(
        [
            json.dumps(
                {
                    "themes": [
                        {
                            "title": "Rust",
                            "description": "Memory safety",
                            "retrieval_query": "rust lifetimes",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "retrospective": [{"title": "Rust", "body": "Lifetimes came up [S1] [S99]."}],
                    "predictions": [
                        {
                            "claim": "It may continue.",
                            "rationale": "Sampled comments mention it [S1].",
                            "uncertainty": "Uncertain.",
                        }
                    ],
                }
            ),
        ]
    )
    with ChunkStore(settings.chroma_path) as store:
        store.upsert_chunks(
            [_chunk(1, 10, "Rust lifetimes are still painful", "Rust")],
            [[1.0, 0.0, 0.0]],
            "fake-embed",
        )
    with pytest.raises(CitationValidationError, match="not saved"):
        run_generate_report(
            settings=settings,
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            embedder=FakeEmbedder(),
            writer=writer,
        )
    assert json.loads(report_file.read_text(encoding="utf-8")) == {"kind": "prior"}


def test_report_instructions_scope_claims_to_sampled_evidence() -> None:
    text = " ".join(REPORT_INSTRUCTIONS.split())
    assert "several sampled posts/comments" in text
    assert "within the sampled discussions" in text
    assert "the sample included" in text
    assert "strong focus" in text
    assert "Hacker News strongly prefers" in text
    assert "strong demand" in text
    assert "thread title may be described as a topic" in text.lower()
    assert "retrieved comment" in text
    assert "unless a statistic was actually measured" in text


def test_unmeasured_strength_language_is_rejected() -> None:
    from app.report import ThemeSection

    clean = [
        ThemeSection(
            title="Hiring",
            body="Several sampled posts mentioned Python and remote work [S1].",
        )
    ]
    reject_unmeasured_strength(clean)
    assert unmeasured_strength_phrases(clean[0].body) == []
    with pytest.raises(UnmeasuredStrengthError, match="strong focus") as caught:
        reject_unmeasured_strength(
            [
                ThemeSection(
                    title="Hiring",
                    body="The sample showed a strong focus on TypeScript [S1].",
                )
            ]
        )
    assert caught.value.phrases == ["strong focus"]

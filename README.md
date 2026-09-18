# Community Voices

A locally runnable web app that produces a **Community Voices** document for [Hacker News](https://news.ycombinator.com/).

It analyzes a **bounded seven-day sample** of HN comments, generates a **RAG-powered** report of what that sample discussed, makes **clearly labeled predictions** for the following week, and compares that report with a **no-RAG baseline** that receives no current-week corpus evidence.

This is a local take-home app. It does not deploy remotely, and it does not claim that next-week predictions have already been validated.

Recorded example output (from a real run, not live data) is in [`examples/`](examples/). Live generation requires **your own** OpenAI API key.

## Architecture

```
Hacker News (Algolia search_by_date)
        ↓
bounded ingestion (fixed UTC window, thread/comment caps)
        ↓
HTML clean / chunk / content hash
        ↓
OpenAI embeddings
        ↓
local Chroma store
        ↓
bounded theme discovery
        ↓
theme-specific retrieval (window filter, per-thread cap)
        ↓
grounded RAG report with application-assigned citations
```

Separately:

```
same task + same model + same window + no corpus evidence
        ↓
no-RAG baseline
        ↓
comparison (grounding, verifiability, latency, writing-model tokens)
```

The web UI is plain HTML/CSS/JS served by FastAPI. There is no React, LangChain, Docker, or extra database.

## Requirements

- **Python 3.12** (tested on **3.12.9**). Any `python3.12` on your PATH is fine; do not copy a machine-specific Homebrew path.
- An **OpenAI API key** with billing/credits for embeddings (`text-embedding-3-small`) and the writing model (`gpt-4.1-mini` by default).
- Local disk only. The app binds to `127.0.0.1` and does not authenticate users.

The homepage and `GET /api/health` start **without** a key. Ingestion, retrieval smoke, RAG generation, and the no-RAG comparison call OpenAI and need a key.

## Setup from a clean clone

```bash
cd community-voices
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` and set your own key:

```
OPENAI_API_KEY=your-key-here
```

Do not commit `.env`. `requirements-dev.txt` includes the runtime packages plus pytest and Ruff. For a run-only install you can use `pip install -r requirements.txt` instead.

## Recommended happy path

For a **fresh current-week run**, you can omit `--start` and `--end`. In that case, ingestion uses the configured rolling seven-day lookback (`LOOKBACK_DAYS`, default 7), and the later report/comparison commands reuse the resulting corpus window.

```bash
python -m app.source_smoke
python -m app.ingest
python -m app.retrieval_smoke
python -m app.generate_report
python -m app.compare_reports
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

To **reproduce the recorded example in `examples/`**, use the same pinned window for ingest, report, and comparison:

```text
--start 2026-09-10T17:00:00+00:00 --end 2026-09-17T17:00:00+00:00
```

```bash
python -m app.source_smoke

python -m app.ingest \
  --start 2026-09-10T17:00:00+00:00 \
  --end 2026-09-17T17:00:00+00:00

python -m app.retrieval_smoke

python -m app.generate_report \
  --start 2026-09-10T17:00:00+00:00 \
  --end 2026-09-17T17:00:00+00:00

python -m app.compare_reports \
  --start 2026-09-10T17:00:00+00:00 \
  --end 2026-09-17T17:00:00+00:00

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Tests and lint require no internet, no API key, and do not write to `./data/chroma`:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Window behavior:

- `ingest` without `--start` / `--end` uses a rolling lookback from now.
- `generate_report` without explicit dates uses the last ingest window from corpus metadata.
- `compare_reports` without explicit dates uses the window stored on the persisted RAG report and **fails** if that window or generation model does not match.
- Using a pinned window is recommended when you want reproducible reruns or want to reproduce the recorded example artifacts.

## Why pin `--start` / `--end`

A rolling “past 7 days from now” window moves between runs, so a second ingest is not an idempotency test. Pinning a half-open UTC interval `[start, end)` keeps discovery, sampling, retrieval, RAG generation, and the no-RAG baseline on the **same** comments. The ingest CLI prints the exact reuse command after each run.

## Sampling

This is a **bounded sample**, not an archive of all Hacker News.

- One discovery request per lookback day, then at most `MAX_THREADS` threads (default 50) and `MAX_COMMENTS_PER_THREAD` comments per thread (default 12).
- For a fixed window, threads and comments are chosen by **lowest source IDs**, not “newest right now,” so reruns stay stable.
- That ID-stable rule is a reproducibility tradeoff: it can under-represent later comments in the window.
- Each successful ingest **replaces** the Chroma corpus with that sample so leftover threads do not accumulate.
- Report claims are scoped to this sample. Vector distance is nearness only, not popularity, truth, or confidence.

Limits live in `.env` (`LOOKBACK_DAYS`, `MAX_THREADS`, `MAX_COMMENTS_PER_THREAD`, `MAX_HTTP_REQUESTS`, `MAX_INGEST_SECONDS`, `MAX_CHUNK_CHARS`).

## RAG vs no-RAG

Both arms use the **same** generation model, report task, output shape, temperature, and date window.

| | RAG | No-RAG |
|---|---|---|
| Current-week HN comments, titles, themes, overview | yes, retrieved | **none** |
| Browsing / search tools | no | no |
| Citations | application-assigned `[S1]`… mapped to stored HN URLs | no Sources section |
| Latency | **end-to-end** (theme discovery + retrieval + writing) | baseline generation only |
| Tokens | writing-model input/output; **embeddings not included** | writing-model only |

The comparison records grounding, specificity, verifiability, and coverage in prose. It does **not** assign an accuracy score or declare a universal winner. Next-week predictions are not scored; the following week has not been observed.

`compare_reports` reuses `./data/latest_report.json` when the window and model match. It does not silently compare mismatched runs.

## HTTP endpoints

| Path | Purpose |
|---|---|
| `/` | Homepage |
| `GET /api/health` | App up; whether an API key is configured (never returns the key) |
| `GET /api/corpus` | Local chunk count |
| `GET /api/report` | Latest RAG report, or `{ "status": "empty" }` |
| `GET /api/comparison` | Latest RAG vs no-RAG comparison, or empty |

## Recorded examples

[`examples/sample-report.json`](examples/sample-report.json) and [`examples/sample-comparison.json`](examples/sample-comparison.json) are **recorded output from a real local run**. They are not live data and do not include secrets. HN item links are public. To generate a fresh report you still need your own API key.

## Known limitations

- Bounded sample, not all of Hacker News.
- Lowest-ID sampling can bias which comments appear.
- Vector similarity is not popularity or factual confidence.
- One community, one default model, one recorded comparison window.
- Predictions remain unvalidated forecasts until a later week is ingested and reviewed.
- RAG “end-to-end report latency” includes retrieval, not just the writing call.
- Token totals are writing-model usage only.

## Layout (Spring Boot mental model)

| This project | Rough Spring Boot equivalent |
| --- | --- |
| `app/config.py` | `@ConfigurationProperties` |
| `app/main.py` | `@SpringBootApplication` plus a small `@RestController` |
| `app/hn_client.py` | `WebClient` wrapper |
| `app/ingest.py` | batch job |
| `app/store.py` | repository over the local vector store |
| `app/embedder.py` / `app/writer.py` | clients created only when needed |
| `app/generate_report.py` / `app/compare_reports.py` | batch jobs |
| `pytest` | JUnit |

How this was built is in [`WORKFLOW.md`](WORKFLOW.md).

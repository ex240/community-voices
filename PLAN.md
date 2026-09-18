# Plan

Community Voices is a take-home for Transcendent Endeavors. This file separates **assignment requirements** from **our design choices**. Stages 1–5 are implemented.

## Assignment requirements

- Choose an online community with frequent weekly discussion.
- Generate a Community Voices document covering the past week and predicting the next week.
- Use RAG to support report generation.
- Automatically populate a vector data store with community voices.
- Bound source volume so ingestion cannot run away.
- Compare generation without RAG against generation with RAG, and include that comparison.
- Ship a public GitHub repo that reviewers can clone and run from the README.

Suggested, not required: embedding visualization, retrieval statistics, crawlers or agentic ingestion.

## Our design choices

- Community: Hacker News.
- Deliverable: a locally runnable FastAPI web app. The report is a web page.
- Source discovery and bounded comment fetch: Hacker News Algolia API (`search_by_date`). No Firebase `kids` recursion and no nested `/items/:id` tree walk.
- Vector store: Chroma with local persistence. No separate database server.
- Models: official OpenAI Python SDK. Embeddings are used in Stage 2; the writing model is used in Stage 3.
- Frontend: plain HTML/CSS/JS served by FastAPI. No React/Node pipeline.
- No LangChain, LlamaIndex, auth, Docker, Redis, Celery, or extra databases.

## Stage 1 (done)

Local app, health endpoint, safe config, tests, README, and a read-only HN source smoke check.

## Stage 2 (done)

Bounded HN sampling across the lookback window, HTML cleaning, chunking, content hashes, OpenAI embeddings, Chroma persistence, idempotent reruns, ingest stats, and a retrieval smoke check.

## Stage 3 (done)

Bounded corpus overview, candidate theme discovery, theme-specific retrieval with a per-thread cap, citation-safe RAG generation, local report persistence, CLI, and homepage rendering.

## Stage 4 (done)

Same model, window, and report task for a no-RAG baseline versus the RAG report. The baseline receives no retrieved evidence. Comparison records grounding/verifiability observations plus latency and token usage, without fabricated accuracy scores.

## Stage 5 (done)

README sufficient for a clean clone, recorded example artifacts, secret/ignore review, metric labels that match what is actually measured, and a short workflow note. No embedding visualization or agentic ingestion.

# Workflow

This take-home was built incrementally with an AI coding agent in Cursor, with a human reviewing diffs, live output, and citations.

Architecture and stage boundaries were decided conversationally up front (`PLAN.md`). Implementation followed those stages with **explicit stop conditions**: Stage 1 local app and HN smoke; Stage 2 bounded ingest; Stage 3 RAG report; Stage 4 no-RAG comparison; Stage 5 submission hardening. Suggested extras (embedding visualizations, agentic crawlers) were left out.

Each stage ran automated tests (no live OpenAI in pytest), then a **live verification** against Algolia and OpenAI when the stage required it.

A live ingest idempotency check showed large “new/changed” churn that unit tests did not expose. That was a **moving lookback window**, newest-first sampling, and a corpus that never dropped unsampled threads—not unstable hashes. The fix was pinned `--start/--end`, lowest-ID sampling, new/changed/unchanged stats, and **replace-the-sample** corpus semantics.

Generated code was not accepted without review. Citation handling, sample-scoped claims, and the no-RAG isolation tests were tightened after inspecting real reports.

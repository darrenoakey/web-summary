# web-summary

URL summarization service. Fetches web pages, extracts text, sends to LLM for summary.

## Architecture

- **FastAPI + uvicorn** on port 8889, managed by `auto` on this Mac
- **LLM routing**: arbiter (spark GPU, `gpt-oss-20b`) first, Ollama fallback
- **Page fetch**: naive HTTP first, Playwright fallback for JS-rendered pages
- **Concurrency**: `asyncio.to_thread` for blocking pipeline, `_ollama_semaphore` serializes local GPU
- **Cache**: SQLite `summary_cache.db`, 24h TTL, keyed on `url|prompt|word_count`

## Endpoints

- `GET /` — health check
- `POST /` — single URL summary (`{"url": "...", "prompt": "...", "word_count": 200}`)
- `POST /execute-batch` — bulk summarize (engine envelope array, concurrent fetch + async arbiter jobs)

## Key Files

- `src/web_summary.py` — all logic (fetch, parse, LLM, cache, server)
- `src/test_summary.py` — integration tests (real arbiter/Ollama, no mocks)

## Gotchas

- Arbiter `gpt-oss-20b` cold start can take 60s+. Arbiter timeout is `(5s connect, 300s read)`.
- Playwright `sync_playwright()` works inside `asyncio.to_thread` — it gets its own thread with no event loop conflict.
- `summary_cache.db` is gitignored runtime data.

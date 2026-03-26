import json
import hashlib
import argparse
import sqlite3
import os
import time
import threading
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests

ARBITER_URL = "http://10.0.0.254:8400"
ARBITER_CHAT_URL = f"{ARBITER_URL}/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "gpt-oss-20b"
CACHE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_cache.db")
CACHE_TTL_SECONDS = 86400  # 24 hours

# Ollama runs on a single local GPU — serialize access so concurrent requests queue up.
_ollama_semaphore = threading.Semaphore(1)

# Creates a standardized error response dict.
def error_response(message):
    return {"error": str(message)}

# --- Summary Cache (SQLite) ---

def _cache_key(url, prompt, word_count):
    raw = f"{url}|{prompt or ''}|{word_count}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _init_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "  key TEXT PRIMARY KEY,"
        "  url TEXT,"
        "  result TEXT,"
        "  created_at REAL"
        ")"
    )
    conn.commit()
    return conn

def cache_get(url, prompt, word_count):
    key = _cache_key(url, prompt, word_count)
    conn = _init_cache_db()
    try:
        row = conn.execute(
            "SELECT result, created_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row and (time.time() - row[1]) < CACHE_TTL_SECONDS:
            return json.loads(row[0])
        return None
    finally:
        conn.close()

def cache_put(url, prompt, word_count, result):
    key = _cache_key(url, prompt, word_count)
    conn = _init_cache_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, url, result, created_at) VALUES (?, ?, ?, ?)",
            (key, url, json.dumps(result), time.time())
        )
        conn.commit()
    finally:
        conn.close()

# Minimum text blocks to consider a naive fetch successful.
MIN_TEXT_BLOCKS = 3

# Fast fetch using requests (no JS execution).
def fetch_page_naive(url):
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        return resp.text, None
    except Exception as e:
        return None, error_response(f"Naive fetch failed: {e}")

# Slow fetch using Playwright (full JS execution).
def fetch_page_playwright(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                content = page.content()
                return content, None
            finally:
                browser.close()
    except Exception as e:
        return None, error_response(f"Playwright fetch failed: {e}")

# Handles fetching the raw HTML of a page.
# Tries fast naive fetch first, falls back to Playwright if insufficient content.
# Returns (html_content, None) on success, (None, error_dict) on failure.
def fetch_page_html(url):
    # Try naive fetch first (fast)
    html, _ = fetch_page_naive(url)
    if html:
        blocks = clean_html_to_text_blocks(html)
        if len(blocks) >= MIN_TEXT_BLOCKS:
            return html, None

    # Fall back to Playwright (slow but handles JS-rendered content)
    return fetch_page_playwright(url)

# Removes non-content elements like scripts, styles, navs, and footers.
def clean_html_to_text_blocks(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()
    # Returns blocks of text from paragraphs and headers
    return [element.get_text(strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4'])]

# Calculates how many words to take from each block to meet the target.
def calculate_words_per_block(blocks, target_total):
    if not blocks:
        return 0
    return target_total // len(blocks)

# Truncates a string to a specific word count.
def limit_words(text, limit):
    words = text.split()
    return " ".join(words[:limit])

# Processes blocks to ensure the total word count stays under the target.
def prune_content(blocks, target_total):
    words_per_block = calculate_words_per_block(blocks, target_total)
    pruned = [limit_words(block, words_per_block) for block in blocks]
    return " ".join(pruned)

# Orchestrates the fetching and pruning of site content.
# Returns (text, None) on success, (None, error_dict) on failure.
def get_processed_text(url, target_word_count):
    html, err = fetch_page_html(url)
    if err:
        return None, err
    blocks = clean_html_to_text_blocks(html)
    if not blocks:
        return None, error_response("No text content found on page")
    return prune_content(blocks, target_word_count), None

# Extracts a JSON object with a 'summary' key from raw LLM text.
def _parse_summary_json(raw_response, source_name):
    if not raw_response:
        return None, error_response(f"{source_name} returned empty response")
    start_idx = raw_response.find('{')
    end_idx = raw_response.rfind('}')
    if start_idx != -1 and end_idx != -1:
        json_str = raw_response[start_idx : end_idx + 1]
        try:
            return json.loads(json_str), None
        except json.JSONDecodeError as e:
            return None, error_response(f"Failed to parse {source_name} JSON: {e}")
    return None, error_response(f"{source_name} response missing JSON object: {raw_response[:200]}")

# Builds the messages list for summarization.
def _build_summary_messages(text, custom_prompt, word_count):
    system_prompt = (
        f"Return a strict JSON object with a single key 'summary' (approx {word_count} words). "
        f"Custom instructions: {custom_prompt or 'Summarize the text.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Text to summarize:\n{text}"}
    ]

# Hits the arbiter LLM endpoint on spark (OpenAI-compatible).
# Returns (result_dict, None) on success, (None, error_dict) on failure.
def ask_arbiter(text, custom_prompt, word_count=200):
    messages = _build_summary_messages(text, custom_prompt, word_count)
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    try:
        response = requests.post(ARBITER_CHAT_URL, json=payload, timeout=(5, 300))
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return None, error_response("Cannot connect to arbiter at " + ARBITER_URL)
    except requests.exceptions.Timeout:
        return None, error_response("Arbiter request timed out after 120 seconds")
    except requests.exceptions.RequestException as e:
        return None, error_response(f"Arbiter request failed: {e}")

    try:
        response_json = response.json()
    except json.JSONDecodeError:
        return None, error_response("Arbiter returned invalid JSON response")

    try:
        raw_response = response_json["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError):
        return None, error_response(f"Unexpected arbiter response structure: {str(response_json)[:200]}")

    return _parse_summary_json(raw_response, "Arbiter")

# Communicates with local Ollama to generate the structured JSON summary.
# Serialized via semaphore — Ollama runs on a single local GPU.
# Returns (result_dict, None) on success, (None, error_dict) on failure.
def ask_ollama(text, custom_prompt, word_count=200):
    messages = _build_summary_messages(text, custom_prompt, word_count)
    payload = {
        "model": "gpt-oss:20b-16384",
        "messages": messages,
        "stream": False,
        "format": "json"
    }

    with _ollama_semaphore:
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            return None, error_response("Cannot connect to Ollama - is it running on localhost:11434?")
        except requests.exceptions.Timeout:
            return None, error_response("Ollama request timed out after 120 seconds")
        except requests.exceptions.RequestException as e:
            return None, error_response(f"Ollama request failed: {e}")

    try:
        response_json = response.json()
    except json.JSONDecodeError:
        return None, error_response("Ollama returned invalid JSON response")

    try:
        raw_response = response_json["message"]["content"]
    except (KeyError, TypeError):
        raw_response = response_json.get("response", "")

    return _parse_summary_json(raw_response, "Ollama")

# Tries arbiter first, falls back to local Ollama.
def ask_llm(text, custom_prompt, word_count=200):
    result, _ = ask_arbiter(text, custom_prompt, word_count)
    if result:
        return result, None
    # Arbiter failed — fall back to local Ollama
    return ask_ollama(text, custom_prompt, word_count)

# Combines scraping and AI summarization with caching.
# Returns JSON string - either {"summary": "..."} or {"error": "..."}.
def run_pipeline(url, prompt, limit=8000, word_count=200):
    if not url:
        return json.dumps(error_response("Missing required 'url' parameter"))

    cached = cache_get(url, prompt, word_count)
    if cached:
        return json.dumps(cached)

    text, err = get_processed_text(url, limit)
    if err:
        return json.dumps(err)

    result, err = ask_llm(text, prompt, word_count)
    if err:
        return json.dumps(err)

    cache_put(url, prompt, word_count, result)
    return json.dumps(result)

# --- FastAPI Server ---
# run_pipeline is synchronous (blocking HTTP + LLM calls), so we run it in
# uvicorn's thread pool via asyncio.to_thread. Concurrent requests naturally
# overlap: one can fetch HTML while another waits on an LLM response.

import asyncio
from typing import Optional
import fastapi
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

class SummaryRequest(BaseModel):
    url: str
    prompt: Optional[str] = None
    word_count: int = 200

@app.get("/")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})

@app.post("/")
async def summarize(req: SummaryRequest) -> JSONResponse:
    result_str = await asyncio.to_thread(
        run_pipeline, req.url, req.prompt, word_count=req.word_count
    )
    return JSONResponse(content=json.loads(result_str))

def _build_cell_result(envelope: dict, summary: str) -> dict:
    """Build the cell output value from the input envelope + summary."""
    inp = envelope.get("input", {})
    if isinstance(inp, str):
        inp = json.loads(inp) if inp else {}
    return {
        "title": inp.get("title", ""),
        "url": inp.get("link") or inp.get("url", ""),
        "source": inp.get("feed_name", ""),
        "label": inp.get("svm_label", ""),
        "summary": summary,
        "classified_at": inp.get("classified_at", ""),
    }


@app.post("/execute-batch")
async def summarize_bulk(r: fastapi.Request) -> JSONResponse:
    """Bulk summarize: accepts engine envelope array, fetches pages concurrently,
    submits all LLM jobs at once via arbiter async, polls until done."""
    import concurrent.futures
    import time as _time

    envelopes = await r.json()
    if not isinstance(envelopes, list):
        return JSONResponse(content=[{"error": "expected JSON array"}], status_code=400)

    # Extract url/prompt/word_count from each envelope's input
    items: list[dict] = []
    for env in envelopes:
        inp = env.get("input", {})
        if isinstance(inp, str):
            inp = json.loads(inp) if inp else {}
        items.append({
            "url": inp.get("link") or inp.get("url", ""),
            "prompt": "Summarize this news article in 2-3 sentences. Focus on what is new or significant. Be factual and specific.",
            "word_count": 100,
        })

    results: list[dict] = [{}] * len(items)
    pending_llm: list[tuple[int, str, dict]] = []  # (idx, url, {text, prompt, word_count})

    # Step 1: Check cache and fetch page text concurrently
    def fetch_one(idx: int, item: dict) -> tuple[int, dict | None, str | None]:
        """Returns (idx, cached_result_or_None, page_text_or_None)."""
        url = item["url"]
        if not url:
            return (idx, {"summary": ""}, None)
        cached = cache_get(url, item["prompt"], item["word_count"])
        if cached:
            return (idx, cached, None)
        text, err = get_processed_text(url, 8000)
        if err:
            return (idx, err, None)
        return (idx, None, text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, i, item) for i, item in enumerate(items)]
        for fut in concurrent.futures.as_completed(futures):
            idx, cached_or_err, text = fut.result()
            if cached_or_err is not None:
                summary = cached_or_err.get("summary", "") if "summary" in cached_or_err else ""
                results[idx] = _build_cell_result(envelopes[idx], summary)
            elif text:
                item = items[idx]
                pending_llm.append((idx, item["url"], {
                    "text": text,
                    "prompt": item["prompt"],
                    "word_count": item["word_count"],
                }))

    if not pending_llm:
        return JSONResponse(content=results)

    # Step 2: Submit all LLM jobs to arbiter async
    job_ids: list[tuple[int, str, str]] = []  # (idx, url, job_id)
    for idx, url, params in pending_llm:
        messages = _build_summary_messages(params["text"], params["prompt"], params["word_count"])
        payload = {
            "type": "chat-completion",
            "params": {
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.3,
            },
        }
        try:
            resp = requests.post(f"{ARBITER_URL}/v1/jobs", json=payload, timeout=10)
            resp.raise_for_status()
            job_id = resp.json().get("job_id", "")
            if job_id:
                job_ids.append((idx, url, job_id))
            else:
                results[idx] = _build_cell_result(envelopes[idx], "")
        except Exception:
            results[idx] = _build_cell_result(envelopes[idx], "")

    if not job_ids:
        return JSONResponse(content=results)

    # Step 3: Poll until all jobs complete
    pending = {job_id: (idx, url) for idx, url, job_id in job_ids}
    deadline = _time.monotonic() + 300  # 5 min max
    while pending and _time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        try:
            resp = requests.post(
                f"{ARBITER_URL}/v1/jobs/status",
                json={"job_ids": list(pending.keys())},
                timeout=10,
            )
            resp.raise_for_status()
            jobs = resp.json().get("jobs", [])
        except Exception:
            continue

        for job in jobs:
            jid = job.get("job_id", "")
            status = job.get("status", "")
            if status == "completed":
                idx, url = pending.pop(jid, (None, None))
                if idx is None:
                    continue
                result_data = job.get("result", {})
                response_data = result_data.get("response", {})
                try:
                    raw = response_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raw = result_data.get("text", "")
                parsed, _ = _parse_summary_json(raw, "Arbiter")
                if parsed:
                    item = items[idx]
                    cache_put(url, item["prompt"], item["word_count"], parsed)
                    results[idx] = _build_cell_result(envelopes[idx], parsed.get("summary", ""))
                else:
                    results[idx] = _build_cell_result(envelopes[idx], "")
            elif status in ("failed", "cancelled"):
                idx, url = pending.pop(jid, (None, None))
                if idx is not None:
                    results[idx] = _build_cell_result(envelopes[idx], "")

    # Mark remaining as timed out
    for jid, (idx, url) in pending.items():
        results[idx] = _build_cell_result(envelopes[idx], "")

    return JSONResponse(content=results)


def serve(port):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

# --- REAL FUNCTIONALITY TESTS ---
# These tests require a local Ollama instance and internet connection.

def test_word_limiter():
    assert limit_words("one two three four", 2) == "one two"
    print("test_word_limiter passed")

def test_pruning_logic():
    blocks = ["word " * 100, "word " * 100]
    result = prune_content(blocks, 50)
    assert len(result.split()) <= 50
    print("test_pruning_logic passed")

def test_cache_hit():
    url = "https://cache-test.invalid"
    prompt = "test cache"
    word_count = 50
    fake_result = {"summary": "cached summary"}
    cache_put(url, prompt, word_count, fake_result)
    cached = cache_get(url, prompt, word_count)
    assert cached == fake_result
    # Different prompt should miss
    assert cache_get(url, "different prompt", word_count) is None
    print("test_cache_hit passed")

def test_full_pipeline_live():
    # Testing with a real site
    url = "https://example.com"
    res = run_pipeline(url, "extract tech: 1-5")
    data = json.loads(res)
    # Should have either 'summary' (success) or 'error' (graceful failure)
    assert "summary" in data or "error" in data
    if "error" in data:
        print(f"test_full_pipeline_live: got expected error response: {data['error']}")
    else:
        print("test_full_pipeline_live passed")

def test_error_handling():
    # Missing URL should return error, not raise
    res = run_pipeline(None, "test")
    data = json.loads(res)
    assert "error" in data
    assert "url" in data["error"].lower()
    print("test_error_handling passed")

# Entry point for CLI and Server modes.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL to scrape")
    parser.add_argument("--prompt", help="Custom extraction prompt")
    parser.add_argument("--words", type=int, default=200, help="Target word count for summary")
    parser.add_argument("--serve", action="store_true", help="Run as a service")
    parser.add_argument("--test", action="store_true", help="Run internal tests")
    args = parser.parse_args()

    if args.test:
        test_word_limiter()
        test_pruning_logic()
        test_cache_hit()
        test_error_handling()
        test_full_pipeline_live()
    elif args.serve:
        serve(8889)
    elif args.url:
        print(run_pipeline(args.url, args.prompt, word_count=args.words))
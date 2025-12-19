![](banner.jpg)

# Web Summary

A Python tool that fetches web pages, extracts their content, and generates AI-powered summaries using a local Ollama instance.

## Purpose

Web Summary scrapes any URL, cleans the HTML to extract meaningful text content, and uses a local LLM to generate concise summaries. It can be used as a Python library, command-line tool, or HTTP service.

## Installation

1. Install Python dependencies:
```bash
pip install beautifulsoup4 playwright requests
```

2. Install Playwright browsers:
```bash
playwright install chromium
```

3. Ensure you have Ollama running locally on port 11434 with a compatible model.

## Usage

### As a Python Library

```python
import json
import web_summary

result_json = web_summary.run_pipeline(
    url="https://example.com",
    prompt="Focus on the main purpose of the site.",
    word_count=50
)

data = json.loads(result_json)
print(data["summary"])
```

### As a Command-Line Tool

```bash
# Basic usage
python web_summary.py --url "https://example.com" --prompt "Summarize this page"

# With custom word count
python web_summary.py --url "https://example.com" --prompt "Extract key points" --words 100
```

### As an HTTP Service

Start the server:
```bash
python web_summary.py --serve
```

Make requests to the service:
```bash
curl -X POST http://localhost:8889 \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "prompt": "Summarize briefly", "word_count": 50}'
```

Or from Python:
```python
import requests

response = requests.post("http://localhost:8889", json={
    "url": "https://example.com",
    "prompt": "Keep it brief.",
    "word_count": 30
})
data = response.json()
print(data["summary"])
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--url` | URL to scrape and summarize |
| `--prompt` | Custom instructions for the summary |
| `--words` | Target word count for summary (default: 200) |
| `--serve` | Run as an HTTP service on port 8889 |
| `--test` | Run internal tests |

## Running Tests

Run the test suite:
```bash
python -m unittest test_summary.py
```

Or run internal tests:
```bash
python web_summary.py --test
```
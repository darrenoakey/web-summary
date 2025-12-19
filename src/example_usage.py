import json
import requests
import web_summary

# --- Method 1: Using web_summary.py as a library ---
def call_as_library():
    print("--- Method 1: Direct Library Call ---")
    url = "https://example.com"
    prompt = "Focus on the purpose of the site."
    word_count = 30
    
    # Call the pipeline directly
    result_json = web_summary.run_pipeline(url, prompt, word_count=word_count)
    
    data = json.loads(result_json)
    print(f"Summary: {data['summary']}\n")

# --- Method 2: Calling the Summary Service (if running) ---
# To use this, first run: python3 web_summary.py --serve
def call_as_service():
    print("--- Method 2: HTTP API Call ---")
    api_url = "http://localhost:8889"
    payload = {
        "url": "https://example.com",
        "prompt": "Keep it very brief.",
        "word_count": 20
    }
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Summary: {data['summary']}\n")
    except requests.exceptions.ConnectionError:
        print("Error: Service not running. Start it with 'python3 web_summary.py --serve'")

if __name__ == "__main__":
    # Method 1 always works if web_summary.py is in the same directory
    call_as_library()
    
    # Method 2 works if you have the server running in another terminal
    call_as_service()

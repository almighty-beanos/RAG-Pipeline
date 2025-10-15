import requests
import json
import sys

BASE = "http://localhost:8000"

def crawl(start_url):
    r = requests.post(f"{BASE}/crawl", json={"start_url": start_url})
    print(r.json())

def index():
    r = requests.post(f"{BASE}/index", json={})
    print(r.json())

def ask(q):
    r = requests.post(f"{BASE}/ask", json={"question": q, "top_k": 5})
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "crawl":
        crawl(sys.argv[2])
    elif cmd == "index":
        index()
    elif cmd == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        print("Usage: cli.py [crawl <start_url>|index|ask <question>]")

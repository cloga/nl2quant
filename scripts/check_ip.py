#!/usr/bin/env python
"""Check outbound IP for current Python session (respects HTTP(S)_PROXY)."""
import requests

URLS = [
    "https://api.ipify.org?format=json",
    "https://httpbin.org/ip",
]

def main():
    for url in URLS:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            print(f"OK {url}: {resp.text}")
        except Exception as exc:
            print(f"FAIL {url}: {exc}")

if __name__ == "__main__":
    main()

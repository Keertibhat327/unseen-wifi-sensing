"""Read-only authenticated HTTPS deployment checks. Reads secrets from environment only."""

import argparse
import os
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    url = urlparse(args.url)
    if url.scheme != "https" or not url.hostname or url.username or url.query or url.fragment:
        parser.error("provide an HTTPS base URL with no credentials/query")
    key = os.getenv("READ_API_KEY", "")
    if not key:
        parser.error("READ_API_KEY is required in environment")
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=15, follow_redirects=False) as client:
        assert client.get("/health").status_code == 200, "health failed"
        assert client.get("/api/events").status_code == 401, "anonymous reads must be refused"
        for route in (
            "/api/ready",
            "/api/events?limit=1",
            "/api/rooms/status",
            "/api/alerts/status",
        ):
            response = client.get(route, headers={"X-API-Key": key})
            assert response.status_code == 200, f"readiness/read failed: {route}"
            assert response.headers.get("Cache-Control") == "no-store", "private caching enabled"
    print(
        "PASS: HTTPS health, authentication, readiness and read routes; no writes or alerts sent."
    )


if __name__ == "__main__":
    main()

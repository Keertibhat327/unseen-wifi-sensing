"""Bounded HTTPS retries without blocking frame collection on network backoff."""

import random
import time
from urllib.parse import urlparse

import httpx

from gateway.buffer import Outbox, event_body


class Delivery:
    def __init__(
        self,
        url: str,
        api_key: str,
        allow_local_http: bool = False,
        client: httpx.Client | None = None,
    ):
        target = urlparse(url)
        local = target.hostname in {"localhost", "127.0.0.1", "::1"}
        if target.scheme != "https" and not (
            allow_local_http and local and target.scheme == "http"
        ):
            raise ValueError("HTTPS required; local HTTP needs --allow-local-http")
        if (
            not target.hostname
            or target.username
            or target.password
            or target.query
            or target.fragment
        ):
            raise ValueError("invalid API base URL")
        if not api_key:
            raise ValueError("INGEST_API_KEY is required")
        self.url = url.rstrip("/") + "/api/events"
        self.client = client or httpx.Client(timeout=5, follow_redirects=False)
        self.api_key = api_key

    def flush(self, outbox: Outbox, now: float | None = None) -> int:
        """A 2xx only acknowledges the queue item when the response echoes its UUID."""
        current = time.time() if now is None else now
        delivered = 0
        for event_id, body, attempts in outbox.pending(current):
            permanent = False
            retry_after = 0.0
            try:
                response = self.client.post(
                    self.url, json=event_body(body), headers={"X-API-Key": self.api_key}
                )
                if 200 <= response.status_code < 300:
                    acknowledgement = response.json()
                    if (
                        not isinstance(acknowledgement, dict)
                        or acknowledgement.get("event_id") != event_id
                    ):
                        raise ValueError("API acknowledgement UUID mismatch")
                    outbox.acknowledge(event_id)
                    delivered += 1
                    continue
                permanent = (
                    response.status_code not in (408, 425, 429) and response.status_code < 500
                )
                error = f"HTTP {response.status_code}"
                try:
                    retry_after = min(3600, max(0, float(response.headers.get("Retry-After", 0))))
                except ValueError:
                    pass
            except (httpx.HTTPError, ValueError) as exc:
                error = type(exc).__name__
            delay = max(retry_after, min(300, 2 ** min(attempts, 8)) + random.uniform(0, 1))
            outbox.failed(event_id, attempts + 1, current + delay, error, permanent)
        return delivered

    def close(self) -> None:
        self.client.close()

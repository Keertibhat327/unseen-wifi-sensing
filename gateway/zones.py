"""Read local schedules or refresh API schedules off the acquisition thread."""

import json
import threading
import time
from pathlib import Path

import httpx
from pydantic import TypeAdapter

from contracts.zones import Zone


def parse_zones(value: object) -> list[Zone]:
    zones = TypeAdapter(list[Zone]).validate_python(value)
    if len({zone.zone_id for zone in zones}) != len(zones):
        raise ValueError("zone IDs must be unique")
    return zones


class ZoneFeed:
    def __init__(
        self,
        path: Path | None = None,
        api_url: str | None = None,
        read_key: str = "",
        refresh_seconds: float = 10,
        max_age_seconds: float = 60,
    ):
        if path and api_url:
            raise ValueError("choose either local or API zone configuration")
        if api_url and not read_key:
            raise ValueError("READ_API_KEY is required for --sync-zones")
        self.path, self.api_url, self.read_key = path, api_url, read_key
        self.refresh_seconds, self.max_age_seconds = refresh_seconds, max_age_seconds
        self._zones: list[Zone] = []
        self._last_success: float | None = None
        self._next_refresh = 0.0
        self._lock = threading.Lock()

    def refresh(self, client: httpx.Client, now: float | None = None) -> None:
        """The caller must validate HTTPS before supplying api_url (same as event delivery)."""
        current = time.monotonic() if now is None else now
        if not self.api_url or current < self._next_refresh:
            return
        self._next_refresh = current + self.refresh_seconds
        try:
            response = client.get(
                self.api_url.rstrip("/") + "/api/zones", headers={"X-API-Key": self.read_key}
            )
            response.raise_for_status()
            zones = parse_zones(response.json())
        except (httpx.HTTPError, ValueError):
            return  # Snapshot expires below; old schedules are never trusted indefinitely.
        with self._lock:
            self._zones, self._last_success = zones, current

    def snapshot(self, now: float | None = None) -> tuple[list[Zone], str]:
        if self.path:
            # Atomic file replacement allows offline schedule changes without a restart.
            return parse_zones(json.loads(self.path.read_text(encoding="utf-8"))), "local"
        if not self.api_url:
            return [], "disabled"
        current = time.monotonic() if now is None else now
        with self._lock:
            if self._last_success is None:
                return [], "unavailable"
            if current - self._last_success > self.max_age_seconds:
                return [], "expired"
            return list(self._zones), "synced"

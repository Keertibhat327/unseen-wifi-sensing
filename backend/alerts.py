"""Transactional alert intents and lease-based delivery. Telegram is at-least-once."""

import os
import time
from dataclasses import dataclass
from datetime import UTC
from uuid import uuid4

import httpx
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.db import AlertRow
from contracts.events import Event


@dataclass(frozen=True)
class AlertSettings:
    mode: str = "off"
    max_age_seconds: int = 300
    max_attempts: int = 8
    token: str = ""
    chat_id: str = ""

    def __post_init__(self):
        if self.mode not in {"off", "mock", "telegram"}:
            raise ValueError("ALERT_MODE must be off, mock or telegram")
        if not 1 <= self.max_age_seconds <= 86400 or not 1 <= self.max_attempts <= 20:
            raise ValueError("invalid alert age or attempt limit")
        if self.mode == "telegram" and (not self.token or not self.chat_id):
            raise ValueError("Telegram mode requires bot token and chat ID")

    @classmethod
    def from_env(cls):
        return cls(
            os.getenv("ALERT_MODE", "off"),
            int(os.getenv("ALERT_MAX_AGE_SECONDS", "300")),
            int(os.getenv("ALERT_MAX_ATTEMPTS", "8")),
            os.getenv("TELEGRAM_BOT_TOKEN", ""),
            os.getenv("TELEGRAM_CHAT_ID", ""),
        )


def alert_intent(event: Event, settings: AlertSettings, now: float | None = None):
    current = time.time() if now is None else now
    if settings.mode == "off" or event.type not in {
        "fall_confirmed",
        "zone_entry",
        "tamper_suspected",
    }:
        return None
    age = current - event.ts.timestamp()
    if age < -30 or age > settings.max_age_seconds:
        return None
    if settings.mode == "telegram" and (
        event.payload.get("replay", True) is not False
        or event.payload.get("input_source") != "live_csi"
    ):
        return None  # Live notification requires an explicit verified live-source marker.
    # Do not copy arbitrary payloads, feature arrays, or identity-like free text.
    message = (
        f"CSI Sentinel research alert: {event.type}\nRoom: {event.room_id}\n"
        f"Device: {event.device_id}\nUTC: {event.ts.astimezone(UTC).isoformat()}\n"
        f"Event: {event.event_id}\nUnvalidated detection; check the room."
    )
    return AlertRow(
        event_id=str(event.event_id),
        mode=settings.mode,
        message=message,
        expires_at=event.ts.timestamp() + settings.max_age_seconds,
    )


def deliver_one(
    engine, settings: AlertSettings, client: httpx.Client | None = None, now: float | None = None
) -> str:
    if settings.mode == "off":
        return "disabled"
    current = time.time() if now is None else now
    token = str(uuid4())
    with Session(engine) as db:
        candidate = db.scalar(
            select(AlertRow.event_id)
            .where(
                AlertRow.mode == settings.mode,
                AlertRow.status.in_(["pending", "sending"]),
                AlertRow.next_attempt <= current,
            )
            .order_by(AlertRow.next_attempt, AlertRow.event_id)
            .limit(1)
        )
        if candidate is None:
            return "idle"
        claimed = db.execute(
            update(AlertRow)
            .where(
                AlertRow.event_id == candidate,
                AlertRow.status.in_(["pending", "sending"]),
                AlertRow.next_attempt <= current,
            )
            .values(
                status="sending",
                lease_token=token,
                next_attempt=current + 60,
                attempts=AlertRow.attempts + 1,
            )
        ).rowcount
        db.commit()
        if not claimed:
            return "busy"
        row = db.get(AlertRow, candidate)
        message, attempts, expires = row.message, row.attempts, row.expires_at
    status, error, delay = "sent", None, min(300, 2 ** min(attempts, 8))
    if expires <= current:
        status, error = "expired", "source_event_expired"
    elif attempts > settings.max_attempts:
        status, error = "dead", "attempt_limit"
    elif settings.mode == "mock":
        status = "mock_sent"
    else:
        own_client = client is None
        client = client or httpx.Client(timeout=10, follow_redirects=False)
        try:
            response = client.post(
                f"https://api.telegram.org/bot{settings.token}/sendMessage",
                json={"chat_id": settings.chat_id, "text": message, "protect_content": True},
            )
            body = response.json()
            if response.status_code == 200 and isinstance(body, dict) and body.get("ok") is True:
                if not isinstance(body.get("result"), dict) or "message_id" not in body["result"]:
                    raise ValueError("missing acknowledgement")
            else:
                code = (
                    body.get("error_code", response.status_code)
                    if isinstance(body, dict)
                    else response.status_code
                )
                permanent = 400 <= code < 500 and code not in (408, 429)
                status, error = ("dead" if permanent else "pending"), f"telegram_{code}"
                if code == 429 and isinstance(body, dict):
                    delay = min(
                        86400, max(delay, float(body.get("parameters", {}).get("retry_after", 0)))
                    )
        except (httpx.HTTPError, ValueError, TypeError):
            status, error = "pending", "transport_or_ack_failure"
        finally:
            if own_client:
                client.close()
    if status == "pending" and attempts >= settings.max_attempts:
        status = "dead"
    with Session(engine) as db:
        db.execute(
            update(AlertRow)
            .where(AlertRow.event_id == candidate, AlertRow.lease_token == token)
            .values(status=status, error=error, next_attempt=current + delay, lease_token=None)
        )
        db.commit()
    return status

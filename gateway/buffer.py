"""A local SQLite outbox preserves UUIDs and device sequences across process restarts."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from contracts.events import Event
from detection.core import Detection


class Outbox:
    def __init__(self, path: Path, max_items: int = 100000):
        if max_items < 1:
            raise ValueError("outbox capacity must be positive")
        self.max_items = max_items
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=15)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS counters(device TEXT PRIMARY KEY, seq INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(
                id TEXT PRIMARY KEY, body TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt REAL NOT NULL DEFAULT 0, error TEXT, quarantined INTEGER DEFAULT 0
            );
        """)

    def enqueue(
        self, detection: Detection, device_id: str, room_id: str, ts: datetime | None = None
    ) -> Event:
        """Sequence allocation and event persistence succeed together or both roll back."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] >= self.max_items:
                raise RuntimeError("outbox capacity reached; drain or archive quarantined events")
            row = self.db.execute(
                "SELECT seq FROM counters WHERE device=?", (device_id,)
            ).fetchone()
            seq = row[0] + 1 if row else 0
            event = Event(
                event_id=uuid4(),
                schema_version=1,
                seq=seq,
                sender_version="gateway/0.2.0",
                device_id=device_id,
                room_id=room_id,
                ts=ts or datetime.now(UTC),
                type=detection.type,
                confidence=detection.confidence,
                payload=detection.payload,
            )
            self.db.execute("INSERT OR REPLACE INTO counters VALUES(?,?)", (device_id, seq))
            self.db.execute(
                "INSERT INTO outbox(id,body) VALUES(?,?)",
                (str(event.event_id), event.model_dump_json()),
            )
        return event

    def pending(self, now: float, limit: int = 20) -> list[tuple]:
        return self.db.execute(
            "SELECT id,body,attempts FROM outbox WHERE quarantined=0 AND next_attempt<=? "
            "ORDER BY rowid LIMIT ?",
            (now, limit),
        ).fetchall()

    def acknowledge(self, event_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM outbox WHERE id=?", (event_id,))

    def failed(
        self, event_id: str, attempts: int, next_attempt: float, error: str, permanent: bool = False
    ) -> None:
        with self.db:
            self.db.execute(
                "UPDATE outbox SET attempts=?,next_attempt=?,error=?,quarantined=? WHERE id=?",
                (attempts, next_attempt, error, int(permanent), event_id),
            )

    def counts(self) -> dict[str, int]:
        rows = self.db.execute("SELECT quarantined,COUNT(*) FROM outbox GROUP BY quarantined")
        values = dict(rows)
        return {"pending": values.get(0, 0), "quarantined": values.get(1, 0)}

    def requeue(self) -> None:
        """Explicit operator action after fixing authentication or incompatible events."""
        with self.db:
            self.db.execute("UPDATE outbox SET quarantined=0,next_attempt=0")

    def close(self) -> None:
        self.db.close()


def event_body(text: str) -> dict:
    return json.loads(text)

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.db import Base
from backend.hardening import validate_production
from detection.core import Detection
from gateway.buffer import Outbox
from scripts.prepare_deploy import prepare


def test_limits_readiness_and_future_time(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}", "write", "read")
    with TestClient(app) as client:
        assert client.get("/api/ready").status_code == 401
        assert client.get("/api/ready", headers={"X-API-Key": "read"}).status_code == 503
        Base.metadata.create_all(app.state.engine)
        assert client.get("/api/ready", headers={"X-API-Key": "read"}).status_code == 200
        oversized = client.post("/api/events", content=b"x" * 32769)
        assert oversized.status_code == 413
        assert oversized.headers["Cache-Control"] == "no-store"
        body = dict(
            event_id=str(uuid4()),
            schema_version=1,
            seq=0,
            sender_version="test",
            device_id="d",
            room_id="r",
            ts=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            type="occupancy",
            confidence=0.5,
            payload={},
        )
        assert (
            client.post("/api/events", json=body, headers={"X-API-Key": "write"}).status_code == 422
        )
    app.state.engine.dispose()


def test_production_guards():
    url = "postgresql://user:password@db.example/db?sslmode=require"
    validate_production(url, "a" * 32, "b" * 32, ["https://dashboard.example"])
    for bad_url, key, origin in [
        ("sqlite:///local.db", "a" * 32, "https://ui.example"),
        (url.split("?")[0], "a" * 32, "https://ui.example"),
        (url, "short", "https://ui.example"),
        (url, "a" * 32, "http://ui.example"),
    ]:
        with pytest.raises(RuntimeError):
            validate_production(bad_url, key, "b" * 32, [origin])


def test_outbox_full_does_not_consume_sequence(tmp_path):
    queue = Outbox(tmp_path / "outbox.db", max_items=1)
    first = queue.enqueue(Detection("motion", 0.5, {}), "d", "r")
    with pytest.raises(RuntimeError, match="capacity"):
        queue.enqueue(Detection("motion", 0.5, {}), "d", "r")
    queue.acknowledge(str(first.event_id))
    assert queue.enqueue(Detection("motion", 0.5, {}), "d", "r").seq == first.seq + 1
    queue.close()


def test_deployment_allowlist(tmp_path):
    output = tmp_path / "api"
    prepare(output)
    files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    assert {"app.py", "backend/app.py", "contracts/events.py", "requirements.txt"} <= files
    assert not any(Path(p).parts[0] in {"identity", "data", "ml", "research"} for p in files)
    assert not any(p.endswith(".db") or p == ".env" for p in files)


def test_additive_schema_setup_preserves_events(tmp_path, monkeypatch):
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from backend.db import EventRow, SchemaRevision, ZoneRow, initialize, make_engine

    url = f"sqlite:///{tmp_path / 'existing.db'}"
    engine = make_engine(url)
    EventRow.__table__.create(engine)
    ZoneRow.__table__.create(engine)
    with Session(engine) as db:
        db.add(
            EventRow(
                event_id=str(uuid4()),
                device_id="d",
                room_id="r",
                seq=1,
                ts="2026-10-10T00:00:00+00:00",
                received_at="2026-10-10T00:00:00+00:00",
                type="occupancy",
                confidence=0.5,
                body={},
            )
        )
        db.commit()
    monkeypatch.setenv("DATABASE_URL", url)
    initialize()
    initialize()
    with Session(engine) as db:
        assert len(list(db.scalars(select(EventRow)))) == 1
        assert db.get(SchemaRevision, 1) is not None
    engine.dispose()

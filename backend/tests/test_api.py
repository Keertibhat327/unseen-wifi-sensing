"""Tests focus on delivery correctness, ordering, access control and persistence."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app import create_app
from backend.db import Base, EventRow

WRITE = {"X-API-Key": "write-test-key"}
READ = {"X-API-Key": "read-test-key"}


@pytest.fixture
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}", "write-test-key", "read-test-key")
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        yield client
    app.state.engine.dispose()


def event(**overrides):
    body = {
        "event_id": str(uuid4()),
        "schema_version": 1,
        "seq": 0,
        "sender_version": "test/1",
        "device_id": "device-1",
        "room_id": "studio",
        "ts": datetime.now(UTC).isoformat(),
        "type": "occupancy",
        "confidence": 0.5,
        "payload": {"occupied": True},
    }
    return body | overrides


def test_auth_and_reserved_types(client):
    assert client.post("/api/events", json=event()).status_code == 401
    assert client.get("/api/events").status_code == 401
    assert client.post("/api/events", headers=READ, json=event()).status_code == 401
    for seq, kind in enumerate(("fire_anomaly", "tamper_suspected")):
        assert (
            client.post("/api/events", headers=WRITE, json=event(seq=seq, type=kind)).status_code
            == 201
        )
    assert len(client.get("/api/events", headers=READ).json()) == 2


def test_idempotency_and_conflicts(client):
    body = event()
    assert client.post("/api/events", headers=WRITE, json=body).status_code == 201
    retry = client.post("/api/events", headers=WRITE, json=body)
    assert retry.status_code == 200 and retry.json()["duplicate"]
    assert (
        client.post("/api/events", headers=WRITE, json=body | {"confidence": 0.7}).status_code
        == 409
    )
    assert client.post("/api/events", headers=WRITE, json=event()).status_code == 409
    assert len(client.get("/api/events", headers=READ).json()) == 1


def test_concurrent_duplicates(client):
    body = event()
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(
            pool.map(
                lambda _: client.post("/api/events", headers=WRITE, json=body).status_code, range(8)
            )
        )
    assert codes.count(201) == 1
    assert codes.count(200) == 7
    with Session(client.app.state.engine) as db:
        assert db.scalar(select(func.count()).select_from(EventRow)) == 1


def test_utc_validation_and_out_of_order_state(client):
    old = event(
        seq=0,
        ts=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        payload={"occupied": False},
    )
    new = event(seq=1)
    assert client.post("/api/events", headers=WRITE, json=new).status_code == 201
    assert client.post("/api/events", headers=WRITE, json=old).status_code == 201
    room = client.get("/api/rooms/status", headers=READ).json()[0]
    assert room["payload"]["occupied"] is True
    assert room["stale"] is False
    invalid = event(seq=2, ts="2026-10-09T10:00:00")
    assert client.post("/api/events", headers=WRITE, json=invalid).status_code == 422
    assert (
        client.post("/api/events", headers=WRITE, json=event(seq=2, schema_version=2)).status_code
        == 422
    )
    device = client.get("/api/devices", headers=READ).json()[0]
    assert device["last_seq"] == 1


def test_timezone_equivalent_retry(client):
    body = event(ts="2026-10-09T12:00:00+05:30")
    assert client.post("/api/events", headers=WRITE, json=body).status_code == 201
    body["ts"] = "2026-10-09T06:30:00Z"
    assert client.post("/api/events", headers=WRITE, json=body).status_code == 200


def test_persistence_zones_and_pagination(client):
    zone = {
        "zone_id": "entry",
        "room_id": "studio",
        "armed": True,
        "starts_at": "2026-10-09T12:00:00+05:30",
        "ends_at": "2026-10-09T13:00:00+05:30",
    }
    assert client.post("/api/zones", headers=WRITE, json=zone).status_code == 200
    assert client.get("/api/zones", headers=READ).json()[0]["starts_at"].endswith("Z")
    bad = zone | {"ends_at": "2026-10-08T00:00:00Z"}
    assert client.post("/api/zones", headers=WRITE, json=bad).status_code == 422
    client.post("/api/events", headers=WRITE, json=event())
    assert client.get("/api/events?limit=201", headers=READ).status_code == 422
    replacement = create_app(str(client.app.state.engine.url), "write-test-key", "read-test-key")
    with TestClient(replacement) as another:
        assert len(another.get("/api/events", headers=READ).json()) == 1
    replacement.state.engine.dispose()


def test_production_rejects_sqlite(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="Postgres"):
        create_app("sqlite:///x.db", "write", "read")


def test_weekly_zone_roundtrip_and_invalid_schedule(client):
    zone = {
        "zone_id": "night",
        "room_id": "studio",
        "armed": True,
        "weekly": [{"weekdays": [4], "start": "22:00:00", "end": "06:00:00"}],
    }
    assert client.post("/api/zones", headers=WRITE, json=zone).status_code == 200
    saved = client.get("/api/zones", headers=READ).json()[0]
    assert saved["weekly"] == zone["weekly"]
    zone["weekly"][0]["start"] = "06:00:00"
    assert client.post("/api/zones", headers=WRITE, json=zone).status_code == 422

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.alerts import AlertSettings, alert_intent, deliver_one
from backend.app import create_app
from backend.db import AlertRow, Base, EventRow, make_engine
from contracts.events import Event


def event(**changes):
    return Event(
        event_id=uuid4(),
        schema_version=1,
        seq=0,
        sender_version="test",
        device_id="test",
        room_id="studio",
        ts=datetime.now(UTC),
        type="fall_confirmed",
        confidence=0.5,
        payload=changes.pop("payload", {}),
        **changes,
    )


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'alerts.db'}")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def enqueue(engine, settings, now):
    e = event(payload={"input_source": "live_csi", "replay": False})
    e.ts = datetime.fromtimestamp(now, UTC)
    with Session(engine) as db:
        db.add(alert_intent(e, settings, now))
        db.commit()
    return str(e.event_id)


def test_transactional_idempotency_and_mock_no_network(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_MODE", "mock")
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}", "write", "read")
    Base.metadata.create_all(app.state.engine)
    body = event().model_dump(mode="json")
    with TestClient(app) as client:
        headers = {"X-API-Key": "write"}
        assert client.post("/api/events", json=body, headers=headers).status_code == 201
        assert client.post("/api/events", json=body, headers=headers).status_code == 200
        changed = body | {"event_id": str(uuid4())}
        assert client.post("/api/events", json=changed, headers=headers).status_code == 409
        with Session(app.state.engine) as db:
            assert db.scalar(select(func.count()).select_from(AlertRow)) == 1
            assert db.scalar(select(func.count()).select_from(EventRow)) == 1

        def forbidden(*args, **kwargs):
            raise AssertionError("mock must not create a network client")

        monkeypatch.setattr(httpx, "Client", forbidden)
        assert deliver_one(app.state.engine, AlertSettings(mode="mock")) == "mock_sent"
        assert deliver_one(app.state.engine, AlertSettings(mode="mock")) == "idle"
    app.state.engine.dispose()


def test_live_suppresses_replay_and_stale():
    settings = AlertSettings(mode="telegram", token="test", chat_id="test")
    assert alert_intent(event(), settings) is None
    assert (
        alert_intent(event(payload={"replay": True, "input_source": "live_csi"}), settings) is None
    )
    e = event(payload={"replay": False, "input_source": "live_csi"})
    e.ts -= timedelta(hours=1)
    assert alert_intent(e, settings) is None
    assert alert_intent(event(), AlertSettings()) is None


def test_telegram_retry_after_ack_and_permanent_failure(engine):
    settings = AlertSettings(mode="telegram", token="secret-not-logged", chat_id="123")
    now = 1000.0
    identifier = enqueue(engine, settings, now)
    replies = [
        httpx.Response(
            429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 40}}
        ),
        httpx.Response(200, json={"ok": True, "result": {"message_id": 42}}),
    ]
    calls = []

    def handler(request):
        calls.append(request)
        return replies.pop(0)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert deliver_one(engine, settings, client, now) == "pending"
        assert deliver_one(engine, settings, client, now + 39) == "idle"
        assert deliver_one(engine, settings, client, now + 40) == "sent"
    assert len(calls) == 2
    assert calls[0].url.host == "api.telegram.org"
    with Session(engine) as db:
        assert db.get(AlertRow, identifier).attempts == 2
        assert db.get(AlertRow, identifier).error is None
    enqueue(engine, settings, now)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(403, json={"ok": False, "error_code": 403})
        )
    ) as client:
        assert deliver_one(engine, settings, client, now) == "dead"


def test_claim_concurrency_expiry_and_restart(engine):
    settings = AlertSettings(mode="mock")
    identifier = enqueue(engine, settings, 1000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: deliver_one(engine, settings, now=1001), range(2)))
    assert results.count("mock_sent") == 1
    with Session(engine) as db:
        row = db.get(AlertRow, identifier)
        row.status, row.next_attempt, row.lease_token = "sending", 1050, "crashed-worker"
        db.commit()
    assert deliver_one(engine, settings, now=1049) == "idle"
    assert deliver_one(engine, settings, now=1050) == "mock_sent"
    enqueue(engine, settings, 1000)
    assert deliver_one(engine, settings, now=2000) == "expired"


def test_lost_ack_is_bounded_and_redacted(engine):
    settings = AlertSettings(
        mode="telegram", token="do-not-store-this-token", chat_id="123", max_attempts=2
    )
    identifier = enqueue(engine, settings, 1000)

    def timeout(request):
        raise httpx.ReadTimeout(f"sensitive URL {request.url}", request=request)

    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        assert deliver_one(engine, settings, client, 1000) == "pending"
        assert deliver_one(engine, settings, client, 1003) == "dead"
        assert deliver_one(engine, settings, client, 1010) == "idle"
    with Session(engine) as db:
        assert db.get(AlertRow, identifier).error == "transport_or_ack_failure"

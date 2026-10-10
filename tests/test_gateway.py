"""Lost acknowledgements, offline persistence, permanent errors and HTTPS enforcement."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.db import Base
from detection.core import Detection
from gateway.buffer import Outbox
from gateway.delivery import Delivery


def detection():
    return Detection("occupancy", 0.5, {"occupied": True, "source": "synthetic_demo"})


def test_restart_and_sequence(tmp_path):
    path = tmp_path / "queue.db"
    queue = Outbox(path)
    first = queue.enqueue(detection(), "device", "room")
    queue.close()
    queue = Outbox(path)
    second = queue.enqueue(detection(), "device", "room")
    assert (first.seq, second.seq) == (0, 1)
    assert queue.pending(1)[0][0] == str(first.event_id)
    queue.close()


def test_retry_preserves_uuid_and_quarantines_4xx(tmp_path):
    queue = Outbox(tmp_path / "queue.db")
    event = queue.enqueue(detection(), "device", "room")
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"event_id": str(event.event_id), "duplicate": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    delivery = Delivery("https://example.test", "key", client=client)
    assert delivery.flush(queue, now=0) == 0
    assert queue.counts()["pending"] == 1
    assert delivery.flush(queue, now=1000) == 1
    assert calls[0] == calls[1]
    queue.enqueue(detection(), "device", "room")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401))) as denied:
        Delivery("https://example.test", "key", client=denied).flush(queue, now=1000)
    assert queue.counts() == {"pending": 0, "quarantined": 1}
    queue.requeue()
    assert queue.counts()["pending"] == 1
    delivery.close()
    queue.close()


def test_response_lost_after_commit_is_safe(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}", "write", "read")
    Base.metadata.create_all(app.state.engine)
    queue = Outbox(tmp_path / "queue.db")
    event = queue.enqueue(detection(), "device", "room")
    calls = 0
    with TestClient(app) as api:

        def handler(request):
            nonlocal calls
            response = api.post(
                "/api/events", json=json.loads(request.content), headers={"X-API-Key": "write"}
            )
            calls += 1
            if calls == 1:
                raise httpx.ReadTimeout("ack lost", request=request)
            return httpx.Response(response.status_code, json=response.json())

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            delivery = Delivery("https://example.test", "write", client=http)
            assert delivery.flush(queue, now=0) == 0
            assert delivery.flush(queue, now=1000) == 1
        events = api.get("/api/events", headers={"X-API-Key": "read"}).json()
        assert len(events) == 1 and events[0]["event_id"] == str(event.event_id)
    queue.close()
    app.state.engine.dispose()


@pytest.mark.parametrize("ack", [{}, [], {"event_id": "wrong-id"}])
def test_https_and_acknowledgement(tmp_path, ack):
    with pytest.raises(ValueError, match="HTTPS"):
        Delivery("http://example.test", "key", True)
    queue = Outbox(tmp_path / "queue.db")
    queue.enqueue(detection(), "device", "room")
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ack))
    ) as http:
        delivery = Delivery("http://localhost:8000", "key", True, http)
        assert delivery.flush(queue, now=0) == 0
    assert queue.counts()["pending"] == 1
    queue.close()

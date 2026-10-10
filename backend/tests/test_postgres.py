"""Optional dedicated test database integration; CI provides a disposable Postgres service."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.alerts import AlertSettings, deliver_one
from backend.app import create_app
from backend.db import AlertRow, Base, EventRow


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not configured")
def test_postgres_event_and_alert_atomicity(monkeypatch):
    monkeypatch.setenv("ALERT_MODE", "mock")
    app = create_app(os.environ["TEST_POSTGRES_URL"], "pg-write", "pg-read")
    Base.metadata.create_all(app.state.engine)
    identifier, device = str(uuid4()), str(uuid4())
    body = dict(
        event_id=identifier,
        device_id=device,
        seq=0,
        room_id="pg-test",
        ts=datetime.now(UTC).isoformat(),
        sender_version="integration",
        schema_version=1,
        type="fall_confirmed",
        confidence=0.5,
        payload={},
    )
    try:
        with TestClient(app) as client:

            def send(_):
                return client.post(
                    "/api/events", json=body, headers={"X-API-Key": "pg-write"}
                ).status_code

            with ThreadPoolExecutor(max_workers=2) as pool:
                assert sorted(pool.map(send, range(2))) == [200, 201]
            with Session(app.state.engine) as db:
                assert (
                    db.scalar(
                        select(func.count())
                        .select_from(AlertRow)
                        .where(AlertRow.event_id == identifier)
                    )
                    == 1
                )
            assert deliver_one(app.state.engine, AlertSettings(mode="mock")) == "mock_sent"
            assert client.get("/api/ready", headers={"X-API-Key": "pg-read"}).status_code == 200
    finally:
        with Session(app.state.engine) as db:
            db.execute(delete(AlertRow).where(AlertRow.event_id == identifier))
            db.execute(delete(EventRow).where(EventRow.event_id == identifier))
            db.commit()
        app.state.engine.dispose()

"""Short, polling-friendly handlers with database-enforced delivery semantics."""

import hmac
import os
from datetime import UTC, datetime, timedelta
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.alerts import AlertSettings, alert_intent
from backend.db import AlertRow, EventRow, ZoneRow, make_engine
from backend.hardening import BodyLimitMiddleware, validate_production
from contracts.events import Event, utc_text
from contracts.zones import Zone


def create_app(
    database_url: str | None = None, ingest_key: str | None = None, read_key: str | None = None
) -> FastAPI:
    load_dotenv()
    url = database_url or os.getenv("DATABASE_URL", "sqlite:///./sentinel.db")
    production = os.getenv("APP_ENV") == "production" or bool(os.getenv("VERCEL"))
    writer = ingest_key or os.getenv("INGEST_API_KEY", "")
    reader = read_key or os.getenv("READ_API_KEY", "")
    if not writer or not reader or writer == reader:
        raise RuntimeError("Configure distinct INGEST_API_KEY and READ_API_KEY values")
    origins = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ]
    if production:
        validate_production(url, writer, reader, origins)
    alerts = AlertSettings.from_env()
    engine = make_engine(url)
    app = FastAPI(
        title="CSI Sentinel",
        version="0.5.0",
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
    )
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    app.add_middleware(BodyLimitMiddleware)

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def check_key(expected: str, supplied: str | None) -> None:
        if not supplied or not hmac.compare_digest(expected.encode(), supplied.encode()):
            raise HTTPException(401, "Invalid API key")

    def write_auth(x_api_key: Annotated[str | None, Header()] = None):
        check_key(writer, x_api_key)

    def read_auth(x_api_key: Annotated[str | None, Header()] = None):
        check_key(reader, x_api_key)

    def session():
        with Session(engine) as db:
            yield db

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.5.0"}

    @app.get("/api/ready", dependencies=[Depends(read_auth)])
    def ready(db: Session = Depends(session)):
        try:
            db.execute(text("SELECT event_id FROM events LIMIT 1"))
            db.execute(text("SELECT event_id FROM alerts LIMIT 1"))
            return {"status": "ready", "alert_mode": alerts.mode}
        except SQLAlchemyError:
            raise HTTPException(503, "Database unavailable or schema not initialized") from None

    @app.get("/api/alerts/status", dependencies=[Depends(read_auth)])
    def alert_status(db: Session = Depends(session)):
        return {
            "mode": alerts.mode,
            "counts": {
                f"{m}:{s}": count
                for m, s, count in db.execute(
                    select(AlertRow.mode, AlertRow.status, func.count()).group_by(
                        AlertRow.mode, AlertRow.status
                    )
                )
            },
        }

    @app.post("/api/events", dependencies=[Depends(write_auth)])
    def ingest(event: Event, response: Response, db: Session = Depends(session)):
        if event.ts > datetime.now(UTC) + timedelta(seconds=300):
            raise HTTPException(422, "event timestamp is more than five minutes in the future")
        body = event.model_dump(mode="json")
        # Canonical UTC representation makes equivalent timezone encodings idempotent.
        body["ts"] = utc_text(event.ts)
        row = EventRow(
            event_id=str(event.event_id),
            device_id=event.device_id,
            room_id=event.room_id,
            seq=event.seq,
            ts=body["ts"],
            received_at=utc_text(datetime.now(UTC)),
            type=event.type,
            confidence=event.confidence,
            body=body,
        )
        db.add(row)
        intent = alert_intent(event, alerts)
        if intent is not None:
            db.add(intent)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = db.get(EventRow, str(event.event_id))
            if existing and existing.body == body:
                return {"event_id": str(event.event_id), "duplicate": True}
            raise HTTPException(
                409, "event_id or device sequence reused with different content"
            ) from None
        response.status_code = 201
        return {"event_id": str(event.event_id), "duplicate": False}

    @app.get("/api/events", dependencies=[Depends(read_auth)])
    def events(
        room_id: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0, le=10000),
        db: Session = Depends(session),
    ):
        query = select(EventRow)
        if room_id:
            query = query.where(EventRow.room_id == room_id)
        rows = db.scalars(
            query.order_by(EventRow.ts.desc(), EventRow.event_id.desc()).offset(offset).limit(limit)
        )
        return [row.body for row in rows]

    @app.get("/api/rooms/status", dependencies=[Depends(read_auth)])
    def rooms(db: Session = Depends(session)):
        # Pick latest source time, not arrival order: a replay cannot roll state backwards.
        latest = (
            select(
                EventRow.room_id,
                EventRow.body,
                EventRow.ts,
                func.row_number()
                .over(
                    partition_by=EventRow.room_id,
                    order_by=(EventRow.ts.desc(), EventRow.seq.desc()),
                )
                .label("rank"),
            )
            .where(EventRow.type.in_(["occupancy", "presence"]))
            .subquery()
        )
        records = db.execute(select(latest).where(latest.c.rank == 1)).mappings()
        stale_before = utc_text(datetime.now(UTC) - timedelta(seconds=30))
        return [
            {
                "room_id": r["room_id"],
                "ts": r["ts"],
                "stale": r["ts"] < stale_before,
                "confidence": r["body"]["confidence"],
                "payload": r["body"]["payload"],
            }
            for r in records
        ]

    @app.get("/api/devices", dependencies=[Depends(read_auth)])
    def devices(db: Session = Depends(session)):
        rows = db.execute(
            select(
                EventRow.device_id,
                func.max(EventRow.ts),
                func.max(EventRow.received_at),
                func.max(EventRow.seq),
            ).group_by(EventRow.device_id)
        )
        cutoff = utc_text(datetime.now(UTC) - timedelta(seconds=30))
        return [
            {
                "device_id": r[0],
                "last_event_at": r[1],
                "last_received_at": r[2],
                "last_seq": r[3],
                "health": "online" if r[1] >= cutoff else "stale",
            }
            for r in rows
        ]

    @app.get("/api/zones", dependencies=[Depends(read_auth)])
    def zones(db: Session = Depends(session)):
        return [row.body for row in db.scalars(select(ZoneRow).order_by(ZoneRow.zone_id))]

    @app.post("/api/zones", dependencies=[Depends(write_auth)])
    def set_zone(zone: Zone, db: Session = Depends(session)):
        db.merge(ZoneRow(zone_id=zone.zone_id, body=zone.model_dump(mode="json")))
        db.commit()
        return zone

    return app

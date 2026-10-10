"""Database tables and explicit schema initialization for M1."""

import os

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    BigInteger,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool


class Base(DeclarativeBase):
    pass


class EventRow(Base):
    __tablename__ = "events"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(96), index=True)
    room_id: Mapped[str] = mapped_column(String(96), index=True)
    seq: Mapped[int] = mapped_column(BigInteger)
    ts: Mapped[str] = mapped_column(String(40), index=True)
    received_at: Mapped[str] = mapped_column(String(40))
    type: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    body: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (
        UniqueConstraint("device_id", "seq", name="uq_device_seq"),
        Index("ix_room_type_ts", "room_id", "type", "ts"),
    )


class ZoneRow(Base):
    __tablename__ = "zones"
    zone_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    body: Mapped[dict] = mapped_column(JSON)


class AlertRow(Base):
    __tablename__ = "alerts"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mode: Mapped[str] = mapped_column(String(12))
    message: Mapped[str] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt: Mapped[float] = mapped_column(Float, default=0)
    expires_at: Mapped[float] = mapped_column(Float)
    # Explicit nullable columns also work with older SQLAlchemy on Python 3.14.
    lease_token = mapped_column(String(36), nullable=True)
    error = mapped_column(String(64), nullable=True)
    __table_args__ = (Index("ix_alert_due", "mode", "status", "next_attempt"),)


class SchemaRevision(Base):
    __tablename__ = "schema_revisions"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)


def make_engine(url: str):
    """Serverless instances avoid retaining an unbounded connection pool."""
    if os.getenv("APP_ENV") == "production" or os.getenv("VERCEL"):
        from urllib.parse import parse_qs, urlparse

        if not url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            raise RuntimeError("Production requires persistent Postgres")
        if parse_qs(urlparse(url).query).get("sslmode", [""])[0] not in {
            "require",
            "verify-ca",
            "verify-full",
        }:
            raise RuntimeError("Production database requires TLS")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(
        url,
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 15}
        if url.startswith("sqlite")
        else {"connect_timeout": 10},
    )


def initialize() -> None:
    """Run once locally or from a trusted migration step, never in request handlers."""
    load_dotenv()
    engine = make_engine(os.environ["DATABASE_URL"])
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import Session

    with Session(engine) as db:
        db.merge(SchemaRevision(version=1))
        db.commit()
    engine.dispose()


if __name__ == "__main__":
    initialize()

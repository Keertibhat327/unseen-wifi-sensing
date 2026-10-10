"""UTC event envelope. Payload additions do not require changing this envelope."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

EventType = Literal[
    "presence",
    "motion",
    "zone_entry",
    "fall_suspected",
    "fall_confirmed",
    "occupancy",
    "fire_anomaly",
    "tamper_suspected",
]


class Event(BaseModel):
    """seq is durable per logical device; retries retain the entire original envelope."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: UUID
    schema_version: Literal[1]
    seq: int = Field(ge=0, le=9_223_372_036_854_775_807)
    sender_version: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=96, pattern=r"^[\w.-]+$")
    room_id: str = Field(min_length=1, max_length=96, pattern=r"^[\w.-]+$")
    ts: datetime
    type: EventType
    confidence: float = Field(ge=0, le=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ts")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        return value.astimezone(UTC)

    @field_validator("payload")
    @classmethod
    def json_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        import json

        forbidden = {
            "identity",
            "identity_prediction",
            "subject",
            "subject_id",
            "person_id",
            "embedding",
            "embeddings",
            "biometric_template",
            "consent_text",
        }

        def reject_identity(item):
            if isinstance(item, dict):
                if any(str(key).casefold() in forbidden for key in item):
                    raise ValueError("identity fields are local-only and forbidden in cloud events")
                for child in item.values():
                    reject_identity(child)
            elif isinstance(item, list):
                for child in item:
                    reject_identity(child)

        reject_identity(value)
        # Strict JSON prevents NaN, hidden binary data, and unbounded event bodies.
        if len(json.dumps(value, allow_nan=False).encode()) > 16_384:
            raise ValueError("payload exceeds 16 KiB")
        return value


def utc_text(value: datetime) -> str:
    """Fixed precision makes UTC strings sortable on both SQLite and Postgres."""
    return value.astimezone(UTC).isoformat(timespec="microseconds")

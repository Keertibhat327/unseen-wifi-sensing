"""UTC room-zone schedules shared by the API and local detector."""

from datetime import UTC, datetime, time, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WeeklySchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weekdays: list[int] = Field(min_length=1, description="UTC start days: Monday=0")
    start: time
    end: time

    @model_validator(mode="after")
    def valid(self) -> "WeeklySchedule":
        if any(day not in range(7) for day in self.weekdays):
            raise ValueError("weekday must be 0..6")
        if self.start.tzinfo or self.end.tzinfo or self.start == self.end:
            raise ValueError("use distinct UTC wall times without an offset")
        return self

    def contains(self, ts: datetime) -> bool:
        current = ts.time().replace(tzinfo=None)
        if self.start < self.end:
            return ts.weekday() in self.weekdays and self.start <= current < self.end
        # An overnight interval belongs to the weekday on which it starts.
        return (ts.weekday() in self.weekdays and current >= self.start) or (
            (ts - timedelta(days=1)).weekday() in self.weekdays and current < self.end
        )


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zone_id: str = Field(min_length=1, max_length=96)
    room_id: str = Field(min_length=1, max_length=96)
    armed: bool = False
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    weekly: list[WeeklySchedule] = Field(default_factory=list)

    @model_validator(mode="after")
    def interval(self) -> "Zone":
        for name in ("starts_at", "ends_at"):
            value = getattr(self, name)
            if value is not None:
                if value.tzinfo is None:
                    raise ValueError("schedule timestamps require UTC offsets")
                setattr(self, name, value.astimezone(UTC))
        if self.starts_at and self.ends_at and self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")
        return self

    def active_at(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            raise ValueError("zone evaluation requires a timezone")
        ts = ts.astimezone(UTC)
        return (
            self.armed
            and (self.starts_at is None or ts >= self.starts_at)
            and (self.ends_at is None or ts < self.ends_at)
            and (not self.weekly or any(schedule.contains(ts) for schedule in self.weekly))
        )

"""Small state machines driven by source-time durations, never replay wall-clock speed."""

from dataclasses import dataclass
from uuid import uuid4

from detection.core import Detection
from detection.settings import DetectorSettings
from detection.signal import Features


@dataclass
class OccupancyState:
    status: str = "unknown"
    active_seconds: float = 0
    empty_seconds: float = 0
    motion_active: bool = False

    def update(
        self, features: Features, duration: float, settings: DetectorSettings, hold_occupied: bool
    ) -> tuple[bool, bool]:
        """Require sustained baseline similarity for vacancy; silence alone is insufficient."""
        previous, previous_motion = self.status, self.motion_active
        if features.motion >= settings.motion_on_sigma:
            self.motion_active = True
        elif features.motion <= settings.motion_off_sigma:
            self.motion_active = False
        evidence = self.motion_active or features.displacement >= settings.presence_on_sigma
        background = (
            features.motion <= settings.motion_off_sigma
            and features.displacement <= settings.presence_off_sigma
        )
        self.active_seconds = self.active_seconds + duration if evidence else 0
        self.empty_seconds = (
            self.empty_seconds + duration if background and not hold_occupied else 0
        )
        if hold_occupied or self.active_seconds >= settings.presence_enter_seconds:
            self.status = "occupied"
        elif self.empty_seconds >= settings.vacancy_seconds:
            self.status = "empty"
        return self.status != previous, self.motion_active != previous_motion


class FallState:
    def __init__(self) -> None:
        self.status = "idle"
        self.incident_id: str | None = None
        self.still_seconds = 0.0
        self.elapsed = 0.0
        self.cooldown = 0.0

    def cancel(self, reason: str, settings: DetectorSettings) -> list[Detection]:
        events = []
        if self.status == "suspected":
            events.append(
                Detection(
                    "fall_suspected",
                    0.5,
                    {
                        "status": "cancelled",
                        "reason": reason,
                        "incident_id": self.incident_id,
                    },
                )
            )
        self.status = "idle"
        self.still_seconds = self.elapsed = 0
        self.cooldown = settings.fall_cooldown_seconds
        return events

    def update(
        self, features: Features, duration: float, settings: DetectorSettings
    ) -> list[Detection]:
        self.cooldown = max(0, self.cooldown - duration)
        if self.status == "confirmed":
            if features.motion >= settings.motion_on_sigma:
                self.cancel("movement_resumed", settings)
            return []
        if self.status == "suspected":
            self.elapsed += duration
            if features.motion >= settings.motion_on_sigma:
                return self.cancel("movement_resumed", settings)
            if features.motion <= settings.fall_stillness_sigma:
                self.still_seconds += duration
            else:
                self.still_seconds = 0
            # A long window arriving beyond the deadline must not confirm retrospectively.
            if self.elapsed > settings.fall_timeout_seconds:
                return self.cancel("stillness_timeout", settings)
            if self.still_seconds >= settings.fall_stillness_seconds:
                self.status = "confirmed"
                return [
                    Detection(
                        "fall_confirmed",
                        0.5,
                        {
                            "status": "confirmed",
                            "incident_id": self.incident_id,
                            "post_fall_stillness_seconds": self.still_seconds,
                        },
                    )
                ]
            if self.elapsed >= settings.fall_timeout_seconds:
                return self.cancel("stillness_timeout", settings)
            return []
        if not self.cooldown and features.impact >= settings.fall_impact_sigma:
            self.status = "suspected"
            self.incident_id = str(uuid4())
            self.still_seconds = self.elapsed = 0
            return [
                Detection(
                    "fall_suspected",
                    0.5,
                    {
                        "status": "suspected",
                        "incident_id": self.incident_id,
                        "impact_score": round(features.impact, 4),
                    },
                )
            ]
        return []

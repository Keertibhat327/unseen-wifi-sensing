"""Calibrated M2 heuristic detector. No networking, source labels or identity features."""

import hashlib
from datetime import UTC, datetime, timedelta

import numpy as np

from contracts.zones import Zone
from detection.adapters import AdapterConfig
from detection.calibration import Calibration
from detection.core import Detection
from detection.settings import DetectorSettings
from detection.signal import extract_features
from detection.states import FallState, OccupancyState


class Detector:
    def __init__(
        self,
        room_id: str,
        config: AdapterConfig,
        calibration: Calibration,
        settings: DetectorSettings,
    ) -> None:
        calibration.validate_for(room_id, config)
        if config.representation != "amplitude":
            raise ValueError("M2 detector requires amplitude; phase inference is not implemented")
        self.room_id, self.config = room_id, config
        self.calibration, self.settings = calibration, settings
        self.occupancy, self.fall = OccupancyState(), FallState()
        self.previous_end: datetime | None = None
        self.calibration_id = calibration.fingerprint()
        self.settings_id = hashlib.sha256(settings.model_dump_json().encode()).hexdigest()

    def process(
        self, values: np.ndarray, ts: datetime, zones: list[Zone] | None = None
    ) -> list[Detection]:
        """ts is the start of a complete, non-overlapping window at the configured cadence."""
        if ts.tzinfo is None:
            raise ValueError("window timestamp requires a timezone")
        ts = ts.astimezone(UTC)
        duration = self.config.window_samples / self.config.sample_rate_hz
        # UTC timestamps round to microseconds; rational sample periods need that tolerance.
        if self.previous_end is not None and ts < self.previous_end - timedelta(microseconds=1):
            raise ValueError("windows must be monotonic and non-overlapping")
        # Validate the entire input before changing any state.
        features = extract_features(values, self.config, self.calibration, self.settings)
        events: list[Detection] = []
        gap = (
            self.previous_end is not None
            and (ts - self.previous_end).total_seconds() > self.settings.gap_tolerance_seconds
        )
        if gap:
            events.extend(self.fall.cancel("data_gap", self.settings))
            self.occupancy = OccupancyState()
        self.previous_end = ts + timedelta(seconds=duration)
        events.extend(self.fall.update(features, duration, self.settings))
        previous_status = self.occupancy.status
        changed, motion_changed = self.occupancy.update(
            features,
            duration,
            self.settings,
            self.fall.status in ("suspected", "confirmed"),
        )
        common = {
            "source": "calibrated_heuristic",
            "validated": False,
            "calibration_id": self.calibration_id,
            "settings_id": self.settings_id,
            "confidence_kind": "uncalibrated_placeholder",
            "occupied": {"occupied": True, "empty": False, "unknown": None}[self.occupancy.status],
            "occupancy_status": self.occupancy.status,
            "count": None,
            "count_supported": False,
            "motion_score": round(features.motion, 4),
            "displacement_score": round(features.displacement, 4),
            "repaired_values": features.repaired_values,
            "data_gap": gap,
            "fall_state": self.fall.status,
        }
        events = [Detection(e.type, e.confidence, common | e.payload) for e in events]
        if motion_changed:
            events.append(
                Detection("motion", 0.5, common | {"active": self.occupancy.motion_active})
            )
        if changed:
            events.append(Detection("presence", 0.5, common))
        if changed and previous_status == "empty" and self.occupancy.status == "occupied":
            for zone in zones or []:
                if zone.room_id == self.room_id and zone.active_at(self.previous_end):
                    events.append(
                        Detection(
                            "zone_entry",
                            0.5,
                            common
                            | {
                                "zone_id": zone.zone_id,
                                "scope": "room",
                                "trigger": "presence_transition",
                            },
                        )
                    )
        # An event each window supplies health even when the occupancy state stays unchanged.
        events.append(Detection("occupancy", 0.5, common))
        return events

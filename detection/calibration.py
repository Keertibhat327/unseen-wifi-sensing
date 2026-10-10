"""Explicit per-room empty-background calibration, stored locally with provenance."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from detection.adapters import AdapterConfig
from detection.core import validate_window
from detection.settings import DetectorSettings


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    room_id: str = Field(min_length=1)
    created_at: datetime
    preprocessing_sha256: str
    baseline: list[float] = Field(min_length=1)
    noise: list[float] = Field(min_length=1)
    window_count: int = Field(ge=2)
    source: str = Field(min_length=1)
    empty_room_asserted: Literal[True] = True

    @field_validator("created_at")
    @classmethod
    def utc_date(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("calibration date requires a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def valid_vectors(self) -> "Calibration":
        if len(self.baseline) != len(self.noise) or min(self.noise) <= 0:
            raise ValueError("calibration vectors must match and noise must be positive")
        return self

    def validate_for(self, room_id: str, config: AdapterConfig) -> None:
        if (
            self.room_id != room_id
            or self.preprocessing_sha256 != config.fingerprint()
            or len(self.baseline) != len(config.subcarrier_indices)
        ):
            raise ValueError("calibration room or preprocessing mismatch; recalibrate explicitly")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def fit_calibration(
    windows: list[np.ndarray],
    room_id: str,
    config: AdapterConfig,
    settings: DetectorSettings,
    source: str,
) -> Calibration:
    """Call only with a known empty-room capture; scene labels are never used to infer it."""
    if config.representation != "amplitude":
        raise ValueError(
            "M2 calibration supports amplitude only; phase needs dedicated preprocessing"
        )
    if len(windows) < settings.calibration_min_windows:
        raise ValueError("not enough empty-room calibration windows")
    for window in windows:
        validate_window(window, config)
    samples = np.concatenate(windows)
    baseline = np.median(samples, axis=0)
    noise = np.maximum(1.4826 * np.median(np.abs(samples - baseline), axis=0), settings.noise_floor)
    drift = max(float(np.median(np.abs(np.median(w, axis=0) - baseline) / noise)) for w in windows)
    if drift > settings.calibration_max_drift_sigma:
        raise ValueError("background changed during calibration; capture a stable empty room")
    return Calibration(
        room_id=room_id,
        created_at=datetime.now(UTC),
        preprocessing_sha256=config.fingerprint(),
        baseline=baseline.tolist(),
        noise=noise.tolist(),
        window_count=len(windows),
        source=source,
    )


def save_calibration(path: Path, calibration: Calibration) -> None:
    """Atomic replacement prevents a partial file from being accepted after power loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(calibration.model_dump(mode="json"), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

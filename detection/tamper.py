"""Local statistical anomaly monitor. It cannot determine intent or prove tampering."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from detection.adapters import AdapterConfig
from detection.core import Detection, validate_window
from ml.artifacts import load_model, save_model


class TamperSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    consecutive_windows: int = Field(default=3, ge=1)
    recovery_windows: int = Field(default=3, ge=1)
    gap_tolerance_seconds: float = Field(default=0.1, ge=0)


def features(x: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            x.mean(axis=(1, 2)),
            x.std(axis=(1, 2)),
            np.abs(np.diff(x, axis=1)).mean(axis=(1, 2)),
            np.abs(x[:, -1] - x[:, 0]).mean(axis=1),
        ],
        axis=1,
    )


def fit_reference(x: np.ndarray) -> dict[str, np.ndarray]:
    if x.ndim != 3 or len(x) < 5 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("at least five finite representative windows required")
    f = features(x)
    center = np.median(f, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(f - center), axis=0), np.maximum(abs(center) * 0.05, 1e-6)
    )
    scores = np.max(np.abs(f - center) / scale, axis=1)
    return {
        "center": center,
        "scale": scale,
        "threshold": np.array([max(6.0, float(np.quantile(scores, 0.99)) * 1.2)]),
    }


def anomaly_scores(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    return np.max(np.abs(features(x) - weights["center"]) / weights["scale"], axis=1)


def save_reference(path: Path, x: np.ndarray, config: AdapterConfig, source: str):
    for window in x:
        validate_window(window, config)
    if config.representation != "amplitude":
        raise ValueError("tamper monitor supports amplitude only")
    save_model(path, fit_reference(x), config, source, "tamper_robust_v1")


class TamperMonitor:
    def __init__(
        self, path: Path, config: AdapterConfig, first: np.ndarray, settings: TamperSettings
    ):
        metadata, weights = load_model(path, config, first, config.sample_rate_hz)
        expected = {"center": (4,), "scale": (4,), "threshold": (1,)}
        if metadata.model_kind != "tamper_robust_v1" or config.representation != "amplitude":
            raise ValueError("not an amplitude anomaly reference")
        if (
            set(weights) != set(expected)
            or any(
                weights[k].shape != shape or not np.isfinite(weights[k]).all()
                for k, shape in expected.items()
            )
            or np.any(weights["scale"] <= 0)
            or weights["threshold"][0] <= 0
        ):
            raise ValueError("invalid anomaly reference")
        self.weights, self.config, self.settings = weights, config, settings
        self.reference_id = metadata.weights_sha256
        self.bad = self.good = 0
        self.active = False
        self.previous_end = None

    def process(self, x: np.ndarray, ts: datetime) -> list[Detection]:
        validate_window(x, self.config)
        if ts.tzinfo is None:
            raise ValueError("timestamp requires timezone")
        ts = ts.astimezone(UTC)
        if self.previous_end is not None:
            gap = (ts - self.previous_end).total_seconds()
            if gap < -1e-6:
                raise ValueError("windows must be monotonic and non-overlapping")
            if gap > self.settings.gap_tolerance_seconds:
                self.bad = self.good = 0
                self.active = False
        self.previous_end = ts + timedelta(
            seconds=self.config.window_samples / self.config.sample_rate_hz
        )
        score = float(anomaly_scores(x[None], self.weights)[0])
        abnormal = score > self.weights["threshold"][0]
        self.bad = self.bad + 1 if abnormal else 0
        self.good = self.good + 1 if not abnormal else 0
        if self.active and self.good >= self.settings.recovery_windows:
            self.active = False
        if self.active or self.bad < self.settings.consecutive_windows:
            return []
        self.active = True
        return [
            Detection(
                "tamper_suspected",
                0.5,
                {
                    "source": "statistical_anomaly",
                    "reason": "sustained_distribution_shift",
                    "score": score,
                    "threshold": float(self.weights["threshold"][0]),
                    "reference_id": self.reference_id,
                    "validated": False,
                    "confidence_kind": "uncalibrated_placeholder",
                    "intent_inferred": False,
                },
            )
        ]

"""Preserve coherent motion while rejecting isolated single-carrier glitches."""

from dataclasses import dataclass

import numpy as np

from detection.adapters import AdapterConfig
from detection.calibration import Calibration
from detection.core import validate_window
from detection.settings import DetectorSettings


@dataclass(frozen=True)
class Features:
    motion: float
    displacement: float
    impact: float
    repaired_values: int


def extract_features(
    values: np.ndarray, config: AdapterConfig, calibration: Calibration, settings: DetectorSettings
) -> Features:
    validate_window(values, config)
    noise = np.asarray(calibration.noise)
    radius = max(1, round(settings.cleaning_window_seconds * config.sample_rate_hz) // 2)
    radius = min(radius, config.window_samples - 1)
    local_median = np.stack(
        [
            np.median(values[max(0, i - radius) : min(len(values), i + radius + 1)], axis=0)
            for i in range(len(values))
        ]
    )
    glitches = np.abs(values - local_median) / noise > settings.outlier_sigma
    coherent = glitches.mean(axis=1) >= settings.coherent_carrier_fraction
    repair = glitches & ~coherent[:, None]
    cleaned = np.where(repair, local_median, values)
    normalized = (cleaned - np.asarray(calibration.baseline)) / noise
    return Features(
        motion=float(np.median(np.std(normalized, axis=0))),
        displacement=float(np.median(np.abs(np.median(normalized, axis=0)))),
        impact=float(np.max(np.median(np.abs(np.diff(normalized, axis=0)), axis=1))),
        repaired_values=int(repair.sum()),
    )

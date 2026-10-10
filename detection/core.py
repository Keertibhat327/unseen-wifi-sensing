"""Shared pure-Python event and window types; inference lives in detection.engine."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from contracts.events import EventType
from detection.adapters.base import AdapterConfig


@dataclass(frozen=True)
class Detection:
    type: EventType
    confidence: float
    payload: dict


def validate_window(values: NDArray, config: AdapterConfig) -> None:
    expected = (config.window_samples, len(config.subcarrier_indices))
    if values.shape != expected or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"expected finite CSI window {expected}, got {values.shape}")

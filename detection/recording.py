"""Read standardized local recordings. Scenario labels are never passed into inference."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from detection.adapters import AdapterConfig
from detection.core import validate_window


@dataclass(frozen=True)
class Window:
    ts: datetime
    values: np.ndarray
    source: str


def read_recording(path: Path, config: AdapterConfig) -> Iterator[Window]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                frame = json.loads(line)
                if frame["sample_rate_hz"] != config.sample_rate_hz:
                    raise ValueError("sample rate differs from config")
                if frame["preprocessing_sha256"] != config.fingerprint():
                    raise ValueError("preprocessing differs from config")
                ts = datetime.fromisoformat(frame["ts"])
                if ts.tzinfo is None:
                    raise ValueError("timestamp must include a UTC offset")
                values = np.asarray(frame["data"], dtype=float)
                validate_window(values, config)
                source = frame.get("source", "recorded_csi")
                if source not in ("synthetic_demo", "recorded_csi"):
                    raise ValueError("unsupported recording source")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid recording at line {number}: {exc}") from exc
            yield Window(ts.astimezone(UTC), values, source)

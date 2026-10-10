"""Explicit representations for an offline privacy attacker, not cloud payloads."""

import numpy as np

from ml.preprocessing import summary_features


def represent(x: np.ndarray, kind: str, factor: int = 4) -> np.ndarray:
    if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("expected finite windows")
    if kind == "raw":
        return x.reshape(len(x), -1)
    if kind == "reduced":
        return summary_features(x)
    if kind == "lower_rate":
        if factor < 2 or x.shape[1] % factor:
            raise ValueError("averaging factor must divide window length and be at least two")
        # Box averaging before decimation; not a hardware anti-aliasing claim.
        return x.reshape(len(x), -1, factor, x.shape[2]).mean(axis=2).reshape(len(x), -1)
    raise ValueError("unknown representation")

"""Per-subcarrier scaling fitted exclusively on the inner training partition."""

import numpy as np


def fit_standardizer(x: np.ndarray) -> dict[str, np.ndarray]:
    if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("expected finite training windows")
    return {
        "mean": x.mean(axis=(0, 1), dtype=np.float64),
        "scale": np.maximum(x.std(axis=(0, 1), dtype=np.float64), 1e-6),
    }


def standardize(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - weights["mean"]) / weights["scale"]).astype(np.float32)


def summary_features(x: np.ndarray) -> np.ndarray:
    """Batch × (four features per carrier): mean, std, range, mean temporal change."""
    return np.concatenate(
        [x.mean(axis=1), x.std(axis=1), np.ptp(x, axis=1), np.abs(np.diff(x, axis=1)).mean(axis=1)],
        axis=1,
    )

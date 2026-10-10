"""Seeded software perturbations of saved amplitude windows; no radio operations."""

import numpy as np


def perturb(x: np.ndarray, kind: str, level: float, scale: np.ndarray, seed: int) -> np.ndarray:
    if x.ndim != 3 or not np.isfinite(x).all() or not np.isfinite(level) or level < 0:
        raise ValueError("invalid perturbation input")
    if scale.shape != (x.shape[2],) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid train-only carrier scale")
    result = x.astype(np.float64, copy=True)
    rng = np.random.default_rng(seed)
    if kind == "clean":
        if level != 0:
            raise ValueError("clean level must be zero")
    elif kind == "noise":
        result += rng.normal(size=x.shape) * scale * level
    elif kind == "drift":
        result += np.linspace(0, level, x.shape[1])[None, :, None] * scale
    elif kind == "packet_loss":
        if level >= 1:
            raise ValueError("packet loss must be below one")
        # Keep first sample; hold last received frame. This models one recovery policy only.
        missing = rng.random(x.shape[:2]) < level
        missing[:, 0] = False
        for t in range(1, x.shape[1]):
            result[missing[:, t], t] = result[missing[:, t], t - 1]
    else:
        raise ValueError("unknown perturbation")
    return np.maximum(result, 0) if kind in ("noise", "drift") else result

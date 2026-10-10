from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from test_ml_data import small_config

from detection.tamper import TamperMonitor, TamperSettings, save_reference
from research.perturbations import perturb
from research.representations import represent


def test_sustained_anomaly_recovery_gap_and_compatibility(tmp_path):
    config = small_config()
    normal = np.random.default_rng(2).normal(10, 0.1, (20, 8, 3))
    path = tmp_path / "normal.csimodel"
    save_reference(path, normal, config, "synthetic test normal")
    monitor = TamperMonitor(
        path, config, normal[0], TamperSettings(consecutive_windows=2, recovery_windows=2)
    )
    ts = datetime.now(UTC)
    abnormal = normal[0] + 1000
    assert not monitor.process(normal[0], ts)
    assert not monitor.process(abnormal, ts + timedelta(seconds=1))
    event = monitor.process(abnormal, ts + timedelta(seconds=2))[0]
    assert event.type == "tamper_suspected" and event.payload["intent_inferred"] is False
    assert not monitor.process(abnormal, ts + timedelta(seconds=3))
    assert not monitor.process(normal[0], ts + timedelta(seconds=4))
    assert not monitor.process(normal[0], ts + timedelta(seconds=5))
    assert not monitor.process(abnormal, ts + timedelta(seconds=6))
    assert not monitor.process(abnormal, ts + timedelta(seconds=10))  # gap resets evidence
    assert monitor.process(abnormal, ts + timedelta(seconds=11))
    with pytest.raises(ValueError, match="monotonic"):
        monitor.process(normal[0], ts)
    with pytest.raises(ValueError):
        TamperMonitor(
            path, config.model_copy(update={"sample_rate_hz": 9}), normal[0], TamperSettings()
        )


def test_perturbation_reproducibility_and_representations():
    x = np.arange(48, dtype=float).reshape(2, 8, 3)
    saved = x.copy()
    scale = np.ones(3)
    for kind, level in (("clean", 0), ("noise", 1), ("drift", 2), ("packet_loss", 0.5)):
        a = perturb(x, kind, level, scale, 4)
        np.testing.assert_array_equal(a, perturb(x, kind, level, scale, 4))
        assert a.shape == x.shape and np.isfinite(a).all()
    np.testing.assert_array_equal(saved, x)
    np.testing.assert_array_equal(perturb(x, "packet_loss", 0, scale, 4), x)
    with pytest.raises(ValueError):
        perturb(x, "packet_loss", 1, scale, 4)
    assert represent(x, "raw").shape == (2, 24)
    assert represent(x, "reduced").shape == (2, 12)
    np.testing.assert_array_equal(
        represent(x, "lower_rate", 4), x.reshape(2, 2, 4, 3).mean(axis=2).reshape(2, -1)
    )

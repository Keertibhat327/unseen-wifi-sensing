"""M2 behavior with synthetic signals; these tests make no real-world accuracy claim."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from contracts.zones import Zone
from detection.adapters import AdapterConfig
from detection.calibration import Calibration, fit_calibration, save_calibration
from detection.engine import Detector
from detection.recording import read_recording
from detection.settings import DetectorSettings
from detection.signal import Features, extract_features
from detection.states import FallState, OccupancyState
from simulator.generate import generate, synthetic_window


@pytest.fixture
def setup():
    config = AdapterConfig.model_validate_json(Path("configs/synthetic.json").read_text())
    settings = DetectorSettings()
    rng = np.random.default_rng(21)
    windows = [synthetic_window(config, "quiet", rng) for _ in range(8)]
    calibration = fit_calibration(windows, "studio", config, settings, "synthetic-test")
    return config, settings, calibration


def sequence(setup, scenarios, zones=None):
    config, settings, calibration = setup
    detector = Detector("studio", config, calibration, settings)
    rng = np.random.default_rng(4)
    start = datetime(2026, 10, 10, tzinfo=UTC)
    duration = config.window_samples / config.sample_rate_hz
    batches = [
        detector.process(
            synthetic_window(config, name, rng), start + timedelta(seconds=i * duration), zones
        )
        for i, name in enumerate(scenarios)
    ]
    return detector, batches


def test_calibration_roundtrip_and_room_config_refusal(setup, tmp_path):
    config, settings, calibration = setup
    path = tmp_path / "calibration.json"
    save_calibration(path, calibration)
    loaded = Calibration.model_validate_json(path.read_text())
    assert loaded == calibration
    with pytest.raises(ValueError, match="mismatch"):
        Detector("another-room", config, loaded, settings)
    with pytest.raises(ValueError, match="mismatch"):
        Detector("studio", config.model_copy(update={"sample_rate_hz": 24}), loaded, settings)
    with pytest.raises(ValueError, match="not enough"):
        fit_calibration([], "studio", config, settings, "test")
    with pytest.raises(ValueError):
        Calibration.model_validate(calibration.model_dump() | {"noise": [0]})


def test_cleaning_rejects_single_carrier_glitch_preserves_coherent_impact(setup):
    config, settings, calibration = setup
    rng = np.random.default_rng(3)
    glitch = extract_features(
        synthetic_window(config, "glitch", rng), config, calibration, settings
    )
    fall = extract_features(synthetic_window(config, "fall", rng), config, calibration, settings)
    assert glitch.repaired_values >= 1
    assert glitch.impact < settings.fall_impact_sigma
    assert fall.impact > settings.fall_impact_sigma


def test_presence_stationary_hold_then_sustained_empty(setup):
    _, batches = sequence(setup, ["quiet"] * 8 + ["motion"] + ["stationary"] * 12 + ["quiet"] * 8)
    assert batches[0][-1].payload["occupancy_status"] == "unknown"
    assert batches[7][-1].payload["occupied"] is False
    assert all(batch[-1].payload["occupied"] is True for batch in batches[8:-1])
    assert batches[-1][-1].payload["occupied"] is False
    assert all(batch[-1].payload["count"] is None for batch in batches)


def test_fall_confirms_once_only_after_post_fall_stillness(setup):
    _, batches = sequence(setup, ["fall"] + ["stillness"] * 12)
    first = next(e for e in batches[0] if e.type == "fall_suspected")
    assert not any(e.type == "fall_confirmed" for e in batches[0] + batches[1])
    confirmations = [e for batch in batches for e in batch if e.type == "fall_confirmed"]
    assert len(confirmations) == 1
    assert confirmations[0].payload["incident_id"] == first.payload["incident_id"]
    assert confirmations[0].payload["post_fall_stillness_seconds"] == 2
    assert batches[-1][-1].payload["occupied"] is True


def test_movement_cancels_fall_and_does_not_confirm_later(setup):
    _, batches = sequence(setup, ["fall", "stillness", "motion"] + ["stillness"] * 5)
    cancelled = [e for batch in batches for e in batch if e.payload.get("status") == "cancelled"]
    assert len(cancelled) == 1 and cancelled[0].payload["reason"] == "movement_resumed"
    assert not any(e.type == "fall_confirmed" for batch in batches for e in batch)


def test_fall_times_out_without_continuous_stillness():
    state, settings = FallState(), DetectorSettings()
    state.update(Features(20, 0, 200, 0), 1, settings)
    events = []
    for _ in range(6):
        events += state.update(Features(5, 0, 0, 0), 1, settings)
    assert len(events) == 1 and events[0].payload["reason"] == "stillness_timeout"
    assert state.status == "idle"


def test_gap_cancels_pending_and_restarts_occupancy_as_unknown(setup):
    config, settings, calibration = setup
    detector = Detector("studio", config, calibration, settings)
    rng = np.random.default_rng(3)
    start = datetime(2026, 10, 10, tzinfo=UTC)
    detector.process(synthetic_window(config, "fall", rng), start)
    events = detector.process(synthetic_window(config, "quiet", rng), start + timedelta(seconds=9))
    assert any(e.payload.get("reason") == "data_gap" for e in events)
    assert events[-1].payload["occupancy_status"] == "unknown"
    assert not any(e.type == "fall_confirmed" for e in events)


def test_bad_windows_do_not_mutate_state(setup):
    config, settings, calibration = setup
    detector = Detector("studio", config, calibration, settings)
    shape = (config.window_samples, len(config.subcarrier_indices))
    start = datetime(2026, 10, 10, tzinfo=UTC)
    with pytest.raises(ValueError):
        detector.process(np.full(shape, np.nan), start)
    assert detector.previous_end is None
    detector.process(np.full(shape, 20.0), start)
    with pytest.raises(ValueError, match="non-overlapping"):
        detector.process(np.full(shape, 20.0), start)
    with pytest.raises(ValueError, match="timezone"):
        detector.process(np.full(shape, 20.0), datetime(2026, 10, 10))


def test_zone_entry_requires_empty_to_occupied_and_matching_armed_room(setup):
    zones = [
        Zone(zone_id="yes", room_id="studio", armed=True),
        Zone(zone_id="no", room_id="other", armed=True),
        Zone(zone_id="off", room_id="studio", armed=False),
    ]
    _, batches = sequence(setup, ["quiet"] * 8 + ["motion"] * 4, zones)
    alerts = [e for batch in batches for e in batch if e.type == "zone_entry"]
    assert len(alerts) == 1 and alerts[0].payload["zone_id"] == "yes"
    _, initial = sequence(setup, ["motion"] * 4, zones)
    assert not any(e.type == "zone_entry" for batch in initial for e in batch)


def test_hysteresis_does_not_chatter():
    state, settings = OccupancyState(), DetectorSettings()
    state.update(Features(9, 0, 0, 0), 1, settings, False)
    state.update(Features(6, 0, 0, 0), 1, settings, False)
    assert state.motion_active and state.status == "occupied"
    state.update(Features(2, 0, 0, 0), 1, settings, False)
    assert not state.motion_active and state.status == "occupied"


def test_alternate_rate_and_shape(setup):
    _, settings, _ = setup
    config = AdapterConfig(
        sample_rate_hz=24,
        window_samples=12,
        subcarrier_indices=list(range(7)),
        pair_order="real_imag",
        ltf_sections=["test"],
        section_byte_ranges={"test": (0, 14)},
    )
    rng = np.random.default_rng(12)
    calibration = fit_calibration(
        [synthetic_window(config, "quiet", rng) for _ in range(8)],
        "studio",
        config,
        settings,
        "test",
    )
    _, batches = sequence((config, settings, calibration), ["fall"] + ["stillness"] * 4)
    assert not any(e.type == "fall_confirmed" for batch in batches[:-1] for e in batch)
    assert any(e.type == "fall_confirmed" for e in batches[-1])


def test_simulator_labels_are_not_used_for_detection(setup, tmp_path):
    config, _, _ = setup
    path = tmp_path / "demo.jsonl"
    generate(path, config, cycles=1)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["scenario"] = "this-label-is-deliberately-wrong"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    _, settings, calibration = setup
    detector = Detector("studio", config, calibration, settings)
    events = [
        e
        for window in read_recording(path, config)
        for e in detector.process(window.values, window.ts)
    ]
    assert sum(e.type == "fall_confirmed" for e in events) == 1
    assert sum(e.payload.get("status") == "cancelled" for e in events) == 1


def test_fractional_window_duration_rounding(setup, tmp_path):
    config, settings, _ = setup
    config = config.model_copy(update={"sample_rate_hz": 60})
    rng = np.random.default_rng(31)
    calibration = fit_calibration(
        [synthetic_window(config, "quiet", rng) for _ in range(8)],
        "studio",
        config,
        settings,
        "test",
    )
    path = tmp_path / "fractional.jsonl"
    generate(path, config, cycles=1, calibration_only=True)
    detector = Detector("studio", config, calibration, settings)
    for window in read_recording(path, config):
        detector.process(window.values, window.ts)


def test_recording_rate_and_preprocessing_are_checked(setup, tmp_path):
    config, _, _ = setup
    path = tmp_path / "bad.jsonl"
    generate(path, config, cycles=1, calibration_only=True)
    row = json.loads(path.read_text().splitlines()[0])
    path.write_text(json.dumps(row | {"sample_rate_hz": 123}))
    with pytest.raises(ValueError, match="sample rate"):
        list(read_recording(path, config))
    path.write_text(json.dumps(row | {"preprocessing_sha256": "incorrect"}))
    with pytest.raises(ValueError, match="preprocessing"):
        list(read_recording(path, config))

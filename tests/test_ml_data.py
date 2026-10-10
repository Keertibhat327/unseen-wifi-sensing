"""Window timing and split isolation are tested independently of optional ML dependencies."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from detection.adapters import AdapterConfig
from ml.data import (
    Recording,
    WindowDataset,
    WindowSettings,
    load_dataset,
    save_dataset,
    window_recording,
)
from ml.preprocessing import fit_standardizer, standardize
from ml.splits import assert_disjoint, make_folds, targets


def small_config() -> AdapterConfig:
    return AdapterConfig(
        sample_rate_hz=8,
        window_samples=8,
        subcarrier_indices=[0, 1, 2],
        pair_order="real_imag",
        ltf_sections=["test"],
        section_byte_ranges={"test": (0, 6)},
    )


def small_dataset() -> WindowDataset:
    rng = np.random.default_rng(8)
    x, actions, people, records, hashes = [], [], [], [], []
    for person in range(3):
        for label in ("fall", "walk", "sit"):
            for recording in range(3):
                identifier = f"ID{person}/{label}/{recording}"
                for _ in range(2):
                    x.append(rng.normal(5 if label == "fall" else 0, 0.2, (8, 3)))
                    actions.append(label)
                    people.append(f"ID{person}")
                    records.append(identifier)
                    hashes.append(hashlib.sha256(identifier.encode()).hexdigest())
    config = small_config()
    manifest = {
        "dataset": "test",
        "archive_sha256": "0" * 64,
        "adapter": config.model_dump(mode="json"),
        "preprocessing_sha256": config.fingerprint(),
        "windowing": {"stride_samples": 8, "max_gap_seconds": 0.2, "duplicate_policy": "mean"},
        "label_scope": "synthetic test only",
        "recordings": [],
    }
    return WindowDataset(
        np.asarray(x, dtype=np.float32),
        np.asarray(actions),
        np.asarray(people),
        np.asarray(records),
        np.asarray(hashes),
        np.arange(len(x), dtype=float),
        manifest,
    )


def test_duplicates_mean_gaps_split_and_no_extrapolation():
    config = small_config()
    times = np.r_[0, 0, np.arange(1, 8) / 8, 3 + np.arange(8) / 8]
    values = np.ones((len(times), 3))
    values[0], values[1] = 2, 4
    recording = Recording("r", "ID0", "walk", times, values, "hash")
    windows, starts, report = window_recording(
        recording, config, WindowSettings(stride_samples=8, max_gap_seconds=0.2)
    )
    assert windows.shape == (2, 8, 3)
    np.testing.assert_array_equal(starts, [0, 3])
    np.testing.assert_array_equal(windows[0, 0], [3, 3, 3])
    assert report["duplicate_timestamps"] == 1 and report["gaps_split"] == 1
    short = Recording("short", "ID0", "walk", np.arange(7) / 8, np.ones((7, 3)), "hash")
    assert not len(
        window_recording(short, config, WindowSettings(stride_samples=8, max_gap_seconds=0.2))[0]
    )


def test_backward_timestamps_rejected():
    r = Recording("r", "ID0", "walk", np.array([1.0, 0.0]), np.ones((2, 3)), "hash")
    with pytest.raises(ValueError, match="timestamps"):
        window_recording(r, small_config(), WindowSettings(stride_samples=8))


def test_lopo_disjoint_recordings_and_train_only_scaling():
    data = small_dataset()
    for fold in make_folds(data, "action", "lopo", 42):
        assert set(data.people[fold.test]) == {fold.name}
        assert fold.name not in data.people[fold.train]
        assert_disjoint(data, fold.train, fold.validation, fold.test)
        scaler = fit_standardizer(data.x[fold.train])
        changed = data.x.copy()
        changed[fold.test] += 10000
        other = fit_standardizer(changed[fold.train])
        np.testing.assert_array_equal(scaler["mean"], other["mean"])
        np.testing.assert_allclose(
            standardize(data.x[fold.train], scaler).mean(axis=(0, 1)), 0, atol=1e-6
        )


def test_duplicate_content_cannot_cross_holdout():
    data = small_dataset()
    data.content_hashes[data.people == "ID1"] = data.content_hashes[0]
    with pytest.raises(ValueError, match="leakage"):
        make_folds(data, "action", "lopo", 42)


def test_loao_reports_untrainable_fall_and_rejects_unseen_multiclass():
    data = small_dataset()
    folds = make_folds(data, "fall", "loao", 42)
    assert next(f for f in folds if f.name == "fall").skip_reason
    for fold in folds:
        assert set(data.actions[fold.test]) == {fold.name}
        if not fold.skip_reason:
            assert fold.name not in data.actions[fold.train]
    with pytest.raises(ValueError, match="unseen"):
        make_folds(data, "action", "loao", 42)
    with pytest.raises(ValueError, match="identity"):
        targets(data, "identity")


def test_dataset_roundtrip_and_missing_class(tmp_path):
    data = small_dataset()
    path = tmp_path / "windows.npz"
    save_dataset(path, data)
    loaded = load_dataset(path)
    np.testing.assert_array_equal(loaded.x, data.x)
    assert loaded.manifest == data.manifest


def test_actual_archives_if_available():
    from ml.data import iter_archive

    config = AdapterConfig.model_validate_json(Path("configs/wifall.json").read_text())
    for dataset, filename in (("wifall", "WiFall.zip"), ("wigesture", "WiGesture.zip")):
        archive = Path(".cache") / filename
        if not archive.exists():
            continue
        recording = next(iter_archive(archive, dataset, config))
        assert recording.person.startswith("ID") and len(recording.values)
        assert recording.values.shape[1] == len(config.subcarrier_indices)

"""Exercise non-default dimensions, decoding choices and artifact refusal paths."""

import json
from pathlib import Path

import numpy as np
import pytest

from detection.adapters import AdapterConfig, WiFallAdapter
from detection.adapters.esp_csi import EspCSIAdapter
from ml.artifacts import load_model, save_model


@pytest.fixture
def config():
    return AdapterConfig(
        sample_rate_hz=24,
        window_samples=12,
        subcarrier_indices=[0, 1],
        pair_order="real_imag",
        ltf_sections=["test"],
        section_byte_ranges={"test": (0, 4)},
    )


def test_complex_decode_and_phase(config):
    row = {"data": "[3,4,5,12]", "len": "4", "first_word_invalid": "0"}
    np.testing.assert_allclose(WiFallAdapter(config).convert(row), [[5, 13]])
    phase = config.model_copy(update={"representation": "phase", "pair_order": "imag_real"})
    np.testing.assert_allclose(
        WiFallAdapter(phase).convert(row), [[np.arctan2(3, 4), np.arctan2(5, 12)]]
    )
    assert WiFallAdapter.timestamp({"timestamp": "2023-11-12 21:21:03.546"}).hour == 13


def test_invalid_bytes_and_sections(config):
    cfg = config.model_copy(
        update={"drop_first_invalid_bytes": True, "section_byte_ranges": {"test": (0, 8)}}
    )
    row = {"data": "[99,99,99,99,3,4,5,12]", "len": "8", "first_word_invalid": "1"}
    np.testing.assert_allclose(WiFallAdapter(cfg).convert(row), [[5, 13]])
    np.testing.assert_allclose(
        WiFallAdapter(cfg).convert(row | {"first_word_invalid": "0"}), [[5, 13]]
    )
    with pytest.raises(ValueError, match="invalid first"):
        WiFallAdapter(config).convert(row)
    with pytest.raises(NotImplementedError, match="TODO"):
        EspCSIAdapter(config).convert("CSI_DATA,...")


def test_models_refuse_changed_shape_rate_preprocessing(config, tmp_path):
    path = tmp_path / "test.csimodel"
    weights = {"coefficients": np.array([0.2, 0.8])}
    metadata = save_model(path, weights, config, "synthetic-test-fixture", "test-only")
    assert metadata.subcarrier_count == 2 and metadata.trained_at.tzinfo is not None
    incoming = np.zeros((12, 2))
    loaded, _ = load_model(path, config, incoming, 24)
    assert loaded == metadata
    for shape, rate in (((12, 3), 24), ((12, 2), 25), ((6, 2), 24)):
        with pytest.raises(ValueError):
            load_model(path, config, np.zeros(shape), rate)
    with pytest.raises(ValueError, match="mismatch"):
        load_model(path, config.model_copy(update={"representation": "phase"}), incoming, 24)
    with pytest.raises(ValueError):
        load_model(path, config, np.full((12, 2), np.nan), 24)


def test_real_wifall_archive_when_available():
    """No dataset is committed or required by CI; locally validate actual downloaded bytes."""
    import csv
    import io
    import zipfile

    path = Path(".cache/WiFall.zip")
    if not path.exists():
        pytest.skip("optional real-data smoke test")
    config = AdapterConfig.model_validate_json(Path("configs/wifall.json").read_text())
    adapter = WiFallAdapter(config)
    with zipfile.ZipFile(path) as archive:
        recording = next(n for n in archive.namelist() if n.endswith(".csv"))
        rows = csv.DictReader(io.TextIOWrapper(archive.open(recording), encoding="utf-8-sig"))
        for _, row in zip(range(config.window_samples), rows, strict=False):
            result = adapter.convert(row)
            assert result.shape == (1, len(config.subcarrier_indices))
            raw = np.array(json.loads(row["data"])).reshape(-1, 2)
            np.testing.assert_allclose(result[0], np.linalg.norm(raw, axis=1))

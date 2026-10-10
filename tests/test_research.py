import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
from test_ml_data import small_dataset  # noqa: E402

from research.privacy import privacy_folds, run  # noqa: E402


def test_privacy_splits_and_complete_experiment(tmp_path):
    data = small_dataset()
    for action, train, test in privacy_folds(data):
        assert all(data.actions[test] == action)
        assert all(data.actions[train] != action)
        assert set(data.people[train]) == set(data.people[test])
        assert not set(data.recordings[train]) & set(data.recordings[test])
    report = run(data, tmp_path / "privacy", factor=4)
    assert len(report["folds"]) == 12
    assert report["uniform_chance"] == pytest.approx(1 / 3)
    assert (tmp_path / "privacy/splits.csv").exists()
    assert len(list(tmp_path.rglob("*.csimodel"))) == 9
    from detection.adapters import AdapterConfig
    from detection.inference import ModelPredictor
    from ml.artifacts import load_model

    config = AdapterConfig.model_validate(data.manifest["adapter"])
    exported = tmp_path / "privacy/fall/raw.csimodel"
    with pytest.raises(ValueError, match="identity"):
        ModelPredictor.load(exported, config, data.x[0], config.sample_rate_hz)
    with pytest.raises(ValueError):
        load_model(exported, config, data.x[0], config.sample_rate_hz + 1)
    data.content_hashes[np.flatnonzero(data.actions == "fall")[0]] = data.content_hashes[
        np.flatnonzero(data.actions == "sit")[0]
    ]
    with pytest.raises(ValueError, match="leakage"):
        list(privacy_folds(data))


def test_robustness_clean_matches_export_and_rejects_provenance(tmp_path):
    pytest.importorskip("torch")
    from ml.data import save_dataset
    from ml.train import TrainingSettings
    from ml.train import run as train
    from research.robustness import run as evaluate

    data_path = tmp_path / "data.npz"
    save_dataset(data_path, small_dataset())
    source = tmp_path / "models"
    train(data_path, source, "fall", "lopo", ["baseline"], TrainingSettings(), ["ID0"])
    settings = {"seed": 42, "conditions": [["clean", 0], ["noise", 1]]}
    output = tmp_path / "robustness"
    evaluate(data_path, source, output, settings)
    report = json.loads((output / "summary.json").read_text())
    original = json.loads((source / "ID0/baseline.metrics.json").read_text())
    assert report["results"][0]["metrics"] == original["window_metrics"]
    assert (output / "ID0/normal-reference.csimodel").exists()
    data = small_dataset()
    data.manifest["windowing"]["stride_samples"] = 4
    save_dataset(data_path, data)
    with pytest.raises(ValueError, match="provenance"):
        evaluate(data_path, source, tmp_path / "refused", settings)

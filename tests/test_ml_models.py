"""Export parity and a complete miniature run; optional training extras required."""

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

from test_ml_data import small_config, small_dataset  # noqa: E402

from detection.inference import ModelPredictor  # noqa: E402
from ml.artifacts import save_model  # noqa: E402
from ml.data import save_dataset  # noqa: E402
from ml.metrics import classification_metrics  # noqa: E402
from ml.models import TinyCNN  # noqa: E402
from ml.preprocessing import fit_standardizer, standardize, summary_features  # noqa: E402
from ml.train import TrainingSettings, run  # noqa: E402


def test_cnn_numpy_export_matches_torch(tmp_path):
    torch.manual_seed(11)
    config = small_config()
    data = small_dataset()
    scaler = fit_standardizer(data.x)
    model = TinyCNN(3, 3, channels=4).eval()
    weights = {key: value.detach().numpy() for key, value in model.state_dict().items()} | scaler
    path = tmp_path / "cnn.csimodel"
    save_model(
        path,
        weights,
        config,
        "synthetic",
        "tiny_cnn_v1",
        ["a", "b", "c"],
        {"task": "action", "channels": 4},
    )
    predictor = ModelPredictor.load(path, config, data.x[0], config.sample_rate_hz)
    with torch.no_grad():
        expected = torch.softmax(
            model(torch.from_numpy(standardize(data.x[:4], scaler))), dim=1
        ).numpy()
    for i in range(4):
        actual = list(predictor.predict(data.x[i], config.sample_rate_hz).probabilities.values())
        np.testing.assert_allclose(actual, expected[i], atol=1e-6)
    with pytest.raises(ValueError, match="mismatch"):
        predictor.predict(data.x[0], 9)
    with pytest.raises(ValueError):
        predictor.predict(data.x[0, :-1], 8)


@pytest.mark.parametrize("binary", [True, False])
def test_sklearn_export_parity(tmp_path, binary):
    from sklearn.linear_model import LogisticRegression

    config, data = small_config(), small_dataset()
    labels = ["nonfall", "fall"] if binary else ["fall", "sit", "walk"]
    y = (
        (data.actions == "fall").astype(int)
        if binary
        else np.array([labels.index(a) for a in data.actions])
    )
    scaler = fit_standardizer(data.x)
    features = summary_features(standardize(data.x, scaler))
    model = LogisticRegression(max_iter=1000).fit(features, y)
    path = tmp_path / "baseline.csimodel"
    save_model(
        path,
        scaler | {"coef": model.coef_, "intercept": model.intercept_},
        config,
        "synthetic",
        "logistic_summary_v1",
        labels,
        {"task": "fall" if binary else "action"},
    )
    predictor = ModelPredictor.load(path, config, data.x[0], 8)
    for i in range(3):
        np.testing.assert_allclose(
            list(predictor.predict(data.x[i], 8).probabilities.values()),
            model.predict_proba(features[i : i + 1])[0],
            atol=1e-6,
        )


def test_end_to_end_training_reports_and_artifacts(tmp_path):
    path = tmp_path / "dataset.npz"
    save_dataset(path, small_dataset())
    output = tmp_path / "run"
    summary = run(
        path,
        output,
        "fall",
        "lopo",
        ["baseline", "cnn"],
        TrainingSettings(cnn_epochs=2, cnn_channels=4),
        ["ID0"],
    )
    assert summary["partial_run"] is True
    for model in ("baseline", "cnn"):
        assert (output / "ID0" / f"{model}.csimodel").exists()
        report = json.loads((output / "ID0" / f"{model}.metrics.json").read_text())
        assert report["window_metrics"]["per_class"]["fall"]["support"] > 0
        assert report["model_metadata"]["training_details"]["outer_fold"] == "ID0"
    assert (output / "ID0" / "splits.csv").exists()
    assert (output / "results.csv").exists()


def test_negative_only_evaluation_does_not_claim_fall_recall():
    report = classification_metrics(np.array([0, 0, 0]), np.array([0, 1, 0]), ["nonfall", "fall"])
    assert report["fall_recall"] is None
    assert report["false_positive_rate"] == pytest.approx(1 / 3)

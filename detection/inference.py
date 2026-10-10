"""Safe exported activity-model inference with NumPy only, including the small CNN."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detection.adapters import AdapterConfig
from detection.core import validate_window
from ml.artifacts import ModelMetadata, load_model
from ml.preprocessing import standardize, summary_features


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    probabilities: dict[str, float]


def softmax(logits: np.ndarray) -> np.ndarray:
    exponent = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Same-padding cross-correlation matching Torch Conv1d for odd kernels."""
    padding = weight.shape[2] // 2
    padded = np.pad(x, ((padding, padding), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, weight.shape[2], axis=0)
    return np.einsum("tck,ock->to", windows, weight, optimize=True) + bias


class ModelPredictor:
    def __init__(
        self, metadata: ModelMetadata, weights: dict[str, np.ndarray], config: AdapterConfig
    ):
        self.metadata, self.weights, self.config = metadata, weights, config
        labels = metadata.class_labels
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("model needs at least two unique activity labels")
        if metadata.inference_config.get("task") not in ("action", "fall"):
            raise ValueError("unsupported task; identity inference is not enabled")
        expected = {"mean": (metadata.subcarrier_count,), "scale": (metadata.subcarrier_count,)}
        if metadata.model_kind == "logistic_summary_v1":
            rows = 1 if len(labels) == 2 else len(labels)
            expected |= {"coef": (rows, 4 * metadata.subcarrier_count), "intercept": (rows,)}
        elif metadata.model_kind == "tiny_cnn_v1":
            channels = metadata.inference_config.get("channels")
            if not isinstance(channels, int) or not 1 <= channels <= 128:
                raise ValueError("invalid CNN width")
            expected |= {
                "conv1.weight": (channels, metadata.subcarrier_count, 5),
                "conv1.bias": (channels,),
                "conv2.weight": (channels * 2, channels, 3),
                "conv2.bias": (channels * 2,),
                "head.weight": (len(labels), channels * 2),
                "head.bias": (len(labels),),
            }
        else:
            raise ValueError("unsupported model kind")
        if (
            set(weights) != set(expected)
            or any(
                weights[key].shape != shape
                or weights[key].dtype.kind not in "if"
                or not np.isfinite(weights[key]).all()
                for key, shape in expected.items()
            )
            or np.any(weights["scale"] <= 0)
        ):
            raise ValueError("invalid model weights or standardizer")

    @classmethod
    def load(
        cls, path: Path, config: AdapterConfig, incoming: np.ndarray, sample_rate_hz: float
    ) -> "ModelPredictor":
        metadata, weights = load_model(path, config, incoming, sample_rate_hz)
        return cls(metadata, weights, config)

    def predict(self, incoming: np.ndarray, sample_rate_hz: float) -> Prediction:
        validate_window(incoming, self.config)
        if (
            sample_rate_hz != self.metadata.sample_rate_hz
            or self.config.fingerprint() != self.metadata.preprocessing_sha256
        ):
            raise ValueError("model/input mismatch")
        x = standardize(incoming, self.weights)
        if not np.isfinite(x).all():
            raise ValueError("standardized input exceeds numeric range")
        if self.metadata.model_kind == "logistic_summary_v1":
            logits = (
                summary_features(x[None])[0] @ self.weights["coef"].T + self.weights["intercept"]
            )
            if len(self.metadata.class_labels) == 2:
                logits = np.array([0.0, logits[0]])
        else:
            x = np.maximum(0, conv1d(x, self.weights["conv1.weight"], self.weights["conv1.bias"]))
            x = np.maximum(0, conv1d(x, self.weights["conv2.weight"], self.weights["conv2.bias"]))
            logits = x.mean(axis=0) @ self.weights["head.weight"].T + self.weights["head.bias"]
        if not np.isfinite(logits).all():
            raise ValueError("model produced non-finite logits")
        probabilities = softmax(logits)
        index = int(probabilities.argmax())
        return Prediction(
            self.metadata.class_labels[index],
            float(probabilities[index]),
            dict(zip(self.metadata.class_labels, map(float, probabilities), strict=True)),
        )

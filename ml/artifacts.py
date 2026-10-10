"""Safe NumPy model bundles, bound to a complete preprocessing fingerprint."""

import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from detection.adapters.base import AdapterConfig
from detection.core import validate_window


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    model_kind: str = Field(min_length=1)
    input_shape: tuple[int, int]
    sample_rate_hz: float = Field(gt=0)
    subcarrier_count: int = Field(gt=0)
    training_dataset: str = Field(min_length=1)
    trained_at: datetime
    preprocessing_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    class_labels: list[str] = Field(default_factory=list)
    inference_config: dict = Field(default_factory=dict)
    training_details: dict = Field(default_factory=dict)

    @field_validator("trained_at")
    @classmethod
    def utc_date(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("training date requires timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def shape_matches(self):
        if self.input_shape[0] < 2 or self.input_shape[1] != self.subcarrier_count:
            raise ValueError("inconsistent model shape")
        return self


def save_model(
    path: Path,
    weights: dict[str, np.ndarray],
    config: AdapterConfig,
    training_dataset: str,
    model_kind: str,
    class_labels: list[str] | None = None,
    inference_config: dict | None = None,
    training_details: dict | None = None,
) -> ModelMetadata:
    """Atomically write one bundle so metadata and weights cannot get separated."""
    if any(np.asarray(v).dtype.kind not in "biufc" for v in weights.values()):
        raise ValueError("only numeric weights supported; pickle is prohibited")
    stream = io.BytesIO()
    np.savez(stream, **weights)
    data = stream.getvalue()
    metadata = ModelMetadata(
        model_kind=model_kind,
        input_shape=(config.window_samples, len(config.subcarrier_indices)),
        sample_rate_hz=config.sample_rate_hz,
        subcarrier_count=len(config.subcarrier_indices),
        training_dataset=training_dataset,
        trained_at=datetime.now(UTC),
        preprocessing_sha256=config.fingerprint(),
        weights_sha256=hashlib.sha256(data).hexdigest(),
        class_labels=class_labels or [],
        inference_config=inference_config or {},
        training_details=training_details or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w") as bundle:
            bundle.writestr("metadata.json", metadata.model_dump_json())
            bundle.writestr("weights.npz", data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata


def load_model(
    path: Path, config: AdapterConfig, incoming: np.ndarray, incoming_sample_rate_hz: float
) -> tuple[ModelMetadata, dict[str, np.ndarray]]:
    """Refuse missing metadata or incompatible input BEFORE exposing any model weights."""
    with zipfile.ZipFile(path) as bundle:
        metadata = ModelMetadata.model_validate(json.loads(bundle.read("metadata.json")))
        validate_window(incoming, config)
        if (
            tuple(incoming.shape) != metadata.input_shape
            or incoming_sample_rate_hz != metadata.sample_rate_hz
            or config.sample_rate_hz != metadata.sample_rate_hz
            or config.fingerprint() != metadata.preprocessing_sha256
        ):
            raise ValueError("model/input mismatch; retrain or explicitly resample and revalidate")
        data = bundle.read("weights.npz")
        if hashlib.sha256(data).hexdigest() != metadata.weights_sha256:
            raise ValueError("model weights checksum mismatch")
        with np.load(io.BytesIO(data), allow_pickle=False) as weights:
            return metadata, {key: weights[key] for key in weights.files}

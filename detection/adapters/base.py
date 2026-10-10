"""Explicit preprocessing configuration prevents silent hardware shape changes."""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample_rate_hz: float = Field(gt=0)
    window_samples: int = Field(ge=2)
    subcarrier_indices: list[int] = Field(min_length=1)
    representation: Literal["amplitude", "phase"] = "amplitude"
    pair_order: Literal["real_imag", "imag_real"]
    drop_first_invalid_bytes: bool = False
    invalid_byte_count: int = Field(default=4, ge=0, multiple_of=2)
    # Byte ranges are in ORIGINAL packet coordinates, before any invalid-byte removal.
    ltf_sections: list[str] = Field(min_length=1)
    section_byte_ranges: dict[str, tuple[int, int]]

    @model_validator(mode="after")
    def valid_layout(self):
        if min(self.subcarrier_indices) < 0 or len(set(self.subcarrier_indices)) != len(
            self.subcarrier_indices
        ):
            raise ValueError("subcarrier indices must be unique and nonnegative")
        if len(set(self.ltf_sections)) != len(self.ltf_sections):
            raise ValueError("LTF section names must be unique")
        for name in self.ltf_sections:
            if name not in self.section_byte_ranges:
                raise ValueError(f"missing byte range for {name}")
            start, end = self.section_byte_ranges[name]
            if start < 0 or end <= start or start % 2 or end % 2:
                raise ValueError("section ranges must contain complete complex pairs")
        return self

    def fingerprint(self) -> str:
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


class CSIAdapter(ABC):
    def __init__(self, config: AdapterConfig):
        self.config = config

    @abstractmethod
    def convert(self, raw_frame: Any) -> NDArray[np.float64]:
        """Convert a frame to (1, configured subcarriers), rejecting unknown layouts."""

    def decode(self, values: list[int], first_word_invalid: bool) -> NDArray[np.float64]:
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.size % 2 or raw.dtype.kind not in "iu":
            raise ValueError("CSI payload must be an even-length integer array")
        if np.any(raw < -128) or np.any(raw > 127):
            raise ValueError("CSI values must be signed bytes")
        if first_word_invalid and not self.config.drop_first_invalid_bytes:
            raise ValueError("invalid first bytes present; configure removal explicitly")
        pieces = []
        for name in self.config.ltf_sections:
            start, end = self.config.section_byte_ranges[name]
            if end > raw.size:
                raise ValueError(f"packet does not contain configured section {name}")
            # Remove the same prefix even on valid packets: feature columns must never shift.
            if self.config.drop_first_invalid_bytes:
                start = max(start, self.config.invalid_byte_count)
            pieces.append(raw[start:end])
        pairs = np.concatenate(pieces).astype(float).reshape(-1, 2)
        if max(self.config.subcarrier_indices) >= len(pairs):
            raise ValueError("configured subcarrier selection exceeds decoded packet")
        pairs = pairs[self.config.subcarrier_indices]
        real, imag = (pairs[:, 0], pairs[:, 1])
        if self.config.pair_order == "imag_real":
            real, imag = imag, real
        values = (
            np.hypot(real, imag)
            if self.config.representation == "amplitude"
            else np.arctan2(imag, real)
        )
        return values.reshape(1, -1)

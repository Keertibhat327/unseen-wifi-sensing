"""Explicit-consent, local numeric templates; no raw windows or network client."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from detection.adapters import AdapterConfig
from detection.core import validate_window
from ml.preprocessing import summary_features

CONSENT_VERSION = "local-identity-research-v1"
CONSENT_TEXT = (
    "I consent to local experimental identity matching of my supplied CSI recordings. "
    "Derived templates are stored on this computer until I delete them. Matching can be wrong; "
    "this is not authentication. My identity and templates must not be sent to the cloud."
)


class IdentityStore:
    def __init__(self, path: Path, *, enabled: bool = False):
        if not enabled:
            raise PermissionError("identity is disabled; explicit local opt-in required")
        if str(path).startswith(("\\\\", "//")):
            raise ValueError("network shares are not supported for local identity storage")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS templates (subject TEXT PRIMARY KEY, record TEXT NOT NULL)"
        )

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def enroll(self, subject: str, windows: np.ndarray, config: AdapterConfig, *, consent: str):
        if consent != CONSENT_TEXT:
            raise PermissionError("the current full consent statement must be explicitly accepted")
        if not subject.strip() or len(subject) > 96:
            raise ValueError("use a short local alias")
        if len(windows) < 5 or config.representation != "amplitude":
            raise ValueError("enrollment needs five amplitude windows from the consenting person")
        for window in windows:
            validate_window(window, config)
        features = summary_features(windows)
        center = features.mean(axis=0)
        scale = np.maximum(features.std(axis=0), np.maximum(abs(center) * 0.05, 1e-6))
        record = {
            "schema_version": 1,
            "consent_version": CONSENT_VERSION,
            "consent_text": consent,
            "consented_at": datetime.now(UTC).isoformat(),
            "input_shape": [config.window_samples, len(config.subcarrier_indices)],
            "sample_rate_hz": config.sample_rate_hz,
            "subcarrier_count": len(config.subcarrier_indices),
            "preprocessing_sha256": config.fingerprint(),
            "training_dataset": "explicit local enrollment",
            "window_count": len(windows),
            "center": center.tolist(),
            "scale": scale.tolist(),
        }
        # No silent replacement of another enrollment or its consent record.
        with self.db:
            self.db.execute(
                "INSERT INTO templates VALUES (?, ?)",
                (subject, json.dumps(record, allow_nan=False)),
            )

    def list_subjects(self) -> list[dict]:
        return [
            {"subject": subject, "consented_at": json.loads(record)["consented_at"]}
            for subject, record in self.db.execute(
                "SELECT subject, record FROM templates ORDER BY subject"
            )
        ]

    def delete(self, subject: str) -> bool:
        with self.db:
            count = self.db.execute("DELETE FROM templates WHERE subject=?", (subject,)).rowcount
        # Consent and derived template share one row; no retained audit identifies the subject.
        self.db.execute("VACUUM")
        return bool(count)

    def match(
        self,
        window: np.ndarray,
        config: AdapterConfig,
        sample_rate_hz: float,
        *,
        max_distance: float = 3.0,
        min_margin: float = 0.5,
    ) -> dict:
        validate_window(window, config)
        if sample_rate_hz != config.sample_rate_hz:
            raise ValueError("incoming sample rate mismatch")
        if not np.isfinite([max_distance, min_margin]).all() or max_distance <= 0 or min_margin < 0:
            raise ValueError("invalid matching thresholds")
        feature = summary_features(window[None])[0]
        distances = []
        for subject, raw in self.db.execute("SELECT subject, record FROM templates"):
            record = json.loads(raw)
            if (
                record["schema_version"] != 1
                or record["consent_version"] != CONSENT_VERSION
                or record["consent_text"] != CONSENT_TEXT
                or record["preprocessing_sha256"] != config.fingerprint()
                or record["sample_rate_hz"] != sample_rate_hz
                or record["input_shape"] != list(window.shape)
            ):
                raise ValueError("enrollment consent or input compatibility mismatch")
            center, scale = np.asarray(record["center"]), np.asarray(record["scale"])
            if (
                center.shape != feature.shape
                or scale.shape != feature.shape
                or not np.isfinite([center, scale]).all()
                or np.any(scale <= 0)
            ):
                raise ValueError("invalid local template")
            distances.append((float(np.sqrt(np.mean(((feature - center) / scale) ** 2))), subject))
        distances.sort()
        result = {"subject": None, "status": "unknown", "validated": False, "local_only": True}
        if distances:
            best, subject = distances[0]
            margin = distances[1][0] - best if len(distances) > 1 else float("inf")
            if best <= max_distance and margin >= min_margin:
                result |= {"subject": subject, "status": "candidate"}
            result["distance"] = best
        return result

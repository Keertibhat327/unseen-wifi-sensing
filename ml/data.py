"""Verified RS2002 archive layouts and leakage-aware, timestamp-based windowing."""

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from detection.adapters import AdapterConfig, WiFallAdapter


class WindowSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    stride_samples: int = Field(gt=0)
    max_gap_seconds: float = Field(default=0.05, gt=0)
    duplicate_policy: str = "mean"


@dataclass
class Recording:
    recording_id: str
    person: str
    action: str
    timestamps: np.ndarray
    values: np.ndarray
    content_hash: str


@dataclass
class WindowDataset:
    x: np.ndarray
    actions: np.ndarray
    people: np.ndarray
    recordings: np.ndarray
    content_hashes: np.ndarray
    starts: np.ndarray
    manifest: dict


def iter_archive(
    path: Path, dataset: str, config: AdapterConfig, exclusions: list[str] | None = None
):
    """Read ZIP members without extracting untrusted paths or loading pickle files."""
    if dataset not in ("wifall", "wigesture"):
        raise ValueError("supported datasets: wifall, wigesture")
    adapter = WiFallAdapter(config)  # Both archives were inspected: same CSV byte encoding.
    with zipfile.ZipFile(path) as archive:
        if set(exclusions or []) - set(archive.namelist()):
            raise ValueError("an excluded recording was not found in the archive")
        for name in sorted(archive.namelist()):
            if name in (exclusions or []):
                continue
            if not name.lower().endswith(".csv") or "__MACOSX" in name:
                continue
            parts = PurePosixPath(name).parts
            if dataset == "wifall" and len(parts) == 4 and parts[0] == "WiFall":
                _, person, action, _ = parts
                action = action.lower()
            elif dataset == "wigesture" and len(parts) == 4 and parts[0] == "dataset":
                _, category, person, filename = parts
                if category not in ("dynamic", "static"):
                    raise ValueError(f"unrecognized WiGesture category: {name}")
                action = f"{category}/{Path(filename).stem.lower()}"
            else:
                raise ValueError(f"unrecognized {dataset} path: {name}")
            if not person.startswith("ID") or not person[2:].isdigit():
                raise ValueError(f"invalid participant directory: {name}")
            values, timestamps = [], []
            with archive.open(name) as raw:
                rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                for row in rows:
                    if row.get("type") != "CSI_DATA":
                        raise ValueError(f"unexpected record type in {name}")
                    expected = action.split("/")[-1]
                    # WiGesture static `taget` is a participant code, not an action label.
                    check_label = dataset == "wifall" or category == "dynamic"
                    if check_label and row.get("taget", "").strip().lower() != expected:
                        raise ValueError(f"row/path action mismatch in {name}")
                    values.append(adapter.convert(row)[0])
                    timestamps.append(adapter.timestamp(row).timestamp())
            if len(values) < 2:
                raise ValueError(f"empty/short recording: {name}")
            array = np.asarray(values, dtype=np.float64)
            yield Recording(
                name,
                person,
                action,
                np.asarray(timestamps),
                array,
                hashlib.sha256(array.tobytes()).hexdigest(),
            )


def window_recording(
    recording: Recording, config: AdapterConfig, settings: WindowSettings
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Average equal-time samples, split gaps, and interpolate without extrapolation."""
    if config.representation != "amplitude" or settings.duplicate_policy != "mean":
        raise ValueError("M3 supports amplitude and mean duplicate aggregation only")
    ts, values = recording.timestamps, recording.values
    if (
        len(ts) != len(values)
        or values.ndim != 2
        or values.shape[1] != len(config.subcarrier_indices)
        or not np.isfinite(values).all()
        or not np.isfinite(ts).all()
        or np.any(np.diff(ts) < 0)
    ):
        raise ValueError(f"invalid timestamps or signal in {recording.recording_id}")
    unique, first, counts = np.unique(ts, return_index=True, return_counts=True)
    values = np.add.reduceat(values, first, axis=0) / counts[:, None]
    dt = np.diff(unique)
    cuts = np.r_[0, np.flatnonzero(dt > settings.max_gap_seconds) + 1, len(unique)]
    windows, starts = [], []
    for left, right in zip(cuts[:-1], cuts[1:], strict=True):
        if right - left < 2:
            continue
        times = unique[left:right] - unique[left]
        count = int(np.floor(times[-1] * config.sample_rate_hz + 1e-5)) + 1
        grid = np.arange(count) / config.sample_rate_hz
        # Drop a rounded endpoint rather than extrapolate across an unobserved time.
        grid = grid[grid <= times[-1] + 1e-8]
        resampled = np.stack(
            [np.interp(grid, times, values[left:right, c]) for c in range(values.shape[1])], axis=1
        )
        for start in range(0, len(grid) - config.window_samples + 1, settings.stride_samples):
            windows.append(resampled[start : start + config.window_samples])
            starts.append(unique[left] + grid[start])
    shape = (0, config.window_samples, len(config.subcarrier_indices))
    x = np.asarray(windows, dtype=np.float32) if windows else np.empty(shape, dtype=np.float32)
    return (
        x,
        np.asarray(starts),
        {
            "recording_id": recording.recording_id,
            "person": recording.person,
            "action": recording.action,
            "rows": len(ts),
            "duplicate_timestamps": int(len(ts) - len(unique)),
            "gaps_split": int(len(cuts) - 2),
            "windows": len(x),
            "observed_median_rate_hz": float(1 / np.median(dt)) if len(dt) else None,
            "duration_seconds": float(ts[-1] - ts[0]),
            "content_sha256": recording.content_hash,
        },
    )


def prepare(
    archive: Path,
    dataset: str,
    config: AdapterConfig,
    settings: WindowSettings,
    exclusions: list[str] | None = None,
) -> WindowDataset:
    features, actions, people, recordings, hashes, starts, reports = [], [], [], [], [], [], []
    for recording in iter_archive(archive, dataset, config, exclusions):
        x, window_starts, report = window_recording(recording, config, settings)
        reports.append(report)
        if not len(x):
            continue
        features.append(x)
        starts.extend(window_starts)
        actions.extend([recording.action] * len(x))
        people.extend([recording.person] * len(x))
        recordings.extend([recording.recording_id] * len(x))
        hashes.extend([recording.content_hash] * len(x))
    if not features:
        raise ValueError("no complete windows survived preparation")
    with archive.open("rb") as handle:
        archive_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "dataset": dataset,
        "archive_sha256": archive_hash,
        "adapter": config.model_dump(mode="json"),
        "preprocessing_sha256": config.fingerprint(),
        "windowing": settings.model_dump(),
        "recordings": reports,
        "explicitly_excluded_recordings": exclusions or [],
        "label_scope": "recording-level; no within-recording fall onset annotations",
    }
    return WindowDataset(
        np.concatenate(features),
        np.asarray(actions),
        np.asarray(people),
        np.asarray(recordings),
        np.asarray(hashes),
        np.asarray(starts),
        manifest,
    )


def save_dataset(path: Path, data: WindowDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=data.x,
        actions=data.actions,
        people=data.people,
        recordings=data.recordings,
        content_hashes=data.content_hashes,
        starts=data.starts,
        manifest=np.array(json.dumps(data.manifest)),
    )


def load_dataset(path: Path) -> WindowDataset:
    with np.load(path, allow_pickle=False) as data:
        result = WindowDataset(
            data["x"],
            data["actions"],
            data["people"],
            data["recordings"],
            data["content_hashes"],
            data["starts"],
            json.loads(str(data["manifest"])),
        )
    config = AdapterConfig.model_validate(result.manifest["adapter"])
    if (
        result.x.ndim != 3
        or tuple(result.x.shape[1:]) != (config.window_samples, len(config.subcarrier_indices))
        or not np.isfinite(result.x).all()
        or not len(result.x)
        or any(
            len(v) != len(result.x)
            for v in (
                result.actions,
                result.people,
                result.recordings,
                result.content_hashes,
                result.starts,
            )
        )
    ):
        raise ValueError("invalid prepared dataset")
    return result

"""Capture an empty-room reference: python -m detection.calibrate --help."""

import argparse
from pathlib import Path

from detection.adapters import AdapterConfig
from detection.calibration import fit_calibration, save_calibration
from detection.recording import read_recording
from detection.settings import DetectorSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.json"))
    parser.add_argument("--settings", type=Path, default=Path("configs/detection.json"))
    parser.add_argument(
        "--confirm-empty-room",
        action="store_true",
        required=True,
        help="Assert this is a known empty, stable room recording",
    )
    args = parser.parse_args()
    config = AdapterConfig.model_validate_json(args.config.read_text())
    settings = DetectorSettings.model_validate_json(args.settings.read_text())
    windows = list(read_recording(args.replay, config))
    duration = config.window_samples / config.sample_rate_hz
    for left, right in zip(windows, windows[1:], strict=False):
        delta = (right.ts - left.ts).total_seconds()
        if delta < duration - 0.000001 or delta - duration > settings.gap_tolerance_seconds:
            raise ValueError("calibration recording has overlaps or missing windows")
    result = fit_calibration(
        [w.values for w in windows], args.room, config, settings, source=str(args.replay)
    )
    save_calibration(args.output, result)
    print(f"Saved {result.window_count}-window calibration for {args.room} to {args.output}")


if __name__ == "__main__":
    main()

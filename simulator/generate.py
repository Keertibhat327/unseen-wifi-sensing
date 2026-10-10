"""Create a repeatable JSONL recording; no cloud keys or networking are needed."""

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from detection.adapters import AdapterConfig


def synthetic_window(config: AdapterConfig, scenario: str, rng: np.random.Generator) -> np.ndarray:
    amplitude = 3.0 if scenario == "motion" else 0.08
    values = 20 + rng.normal(0, amplitude, (config.window_samples, len(config.subcarrier_indices)))
    if scenario in ("stationary", "stillness"):
        values += 2.0
    if scenario == "fall":
        values[config.window_samples // 3] += 35
    if scenario == "glitch":
        values[config.window_samples // 3, 0] += 35
    return values


def generate(
    path: Path,
    config: AdapterConfig,
    cycles: int = 3,
    seed: int = 7,
    calibration_only: bool = False,
) -> None:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    rng = np.random.default_rng(seed)
    duration = config.window_samples / config.sample_rate_hz
    plan = (
        [("quiet", 8)]
        if calibration_only
        else [
            ("quiet", 9),
            ("motion", 2),
            ("stationary", 3),
            ("fall", duration),
            ("stillness", 3),
            ("motion", 4),
            ("quiet", 9),
            ("fall", duration),
            ("motion", 2),
            ("quiet", 9),
        ]
    )
    scenarios = [name for name, seconds in plan for _ in range(math.ceil(seconds / duration))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        start = datetime.now(UTC)
        duration = config.window_samples / config.sample_rate_hz
        for index, scenario in enumerate(scenarios * cycles):
            data = synthetic_window(config, scenario, rng)
            frame = {
                "source": "synthetic_demo",
                "scenario": scenario,
                "ts": (start + timedelta(seconds=index * duration)).isoformat(),
                "sample_rate_hz": config.sample_rate_hz,
                "preprocessing_sha256": config.fingerprint(),
                "data": data.tolist(),
            }
            handle.write(json.dumps(frame) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.json"))
    parser.add_argument("--output", type=Path, default=Path("data/demo.jsonl"))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    generate(
        args.output,
        AdapterConfig.model_validate_json(args.config.read_text()),
        args.cycles,
        calibration_only=args.calibration_only,
    )
    print(args.output)


if __name__ == "__main__":
    main()

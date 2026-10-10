"""Prepare verified CSV archives once; training never reads the evaluation split to scale data."""

import argparse
import json
from pathlib import Path

from detection.adapters import AdapterConfig
from ml.data import WindowSettings, prepare, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset", choices=["wifall", "wigesture"], required=True)
    parser.add_argument("--adapter", type=Path, default=Path("configs/wifall.json"))
    parser.add_argument("--windowing", type=Path, default=Path("configs/windowing.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-recording", action="append", default=[])
    args = parser.parse_args()
    data = prepare(
        args.archive,
        args.dataset,
        AdapterConfig.model_validate_json(args.adapter.read_text()),
        WindowSettings.model_validate_json(args.windowing.read_text()),
        args.exclude_recording,
    )
    save_dataset(args.output, data)
    report = args.output.with_suffix(".manifest.json")
    report.write_text(json.dumps(data.manifest, indent=2), encoding="utf-8")
    print(
        f"Prepared {len(data.x)} windows from {len(data.manifest['recordings'])} recordings: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()

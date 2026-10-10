"""Fit a local anomaly reference from explicitly accepted representative normal replay."""

import argparse
from pathlib import Path

import numpy as np

from detection.adapters import AdapterConfig
from detection.recording import read_recording
from detection.tamper import save_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-representative-normal", action="store_true")
    args = parser.parse_args()
    if not args.confirm_representative_normal:
        parser.error("--confirm-representative-normal required; include expected normal activities")
    config = AdapterConfig.model_validate_json(args.config.read_text())
    x = np.stack([w.values for w in read_recording(args.replay, config)])
    save_reference(args.output, x, config, str(args.replay))


if __name__ == "__main__":
    main()

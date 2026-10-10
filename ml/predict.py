"""Exercise exported NumPy inference on one prepared window, without training dependencies."""

import argparse
import json
from pathlib import Path

from detection.adapters import AdapterConfig
from detection.inference import ModelPredictor
from ml.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    data = load_dataset(args.data)
    if not 0 <= args.index < len(data.x):
        parser.error("index is outside the dataset")
    config = AdapterConfig.model_validate(data.manifest["adapter"])
    window = data.x[args.index]
    model = ModelPredictor.load(args.model, config, window, config.sample_rate_hz)
    prediction = model.predict(window, config.sample_rate_hz)
    print(
        json.dumps(
            {
                "label": prediction.label,
                "score": prediction.score,
                "probabilities": prediction.probabilities,
                "recording_action": str(data.actions[args.index]),
                "note": "Research prediction; check splits.csv for test membership.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

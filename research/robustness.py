"""Evaluate fixed exported M3 models on their original test recordings under perturbations."""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from detection.adapters import AdapterConfig
from detection.inference import ModelPredictor
from detection.tamper import anomaly_scores, fit_reference
from ml.artifacts import save_model
from ml.data import load_dataset
from ml.metrics import classification_metrics
from ml.splits import assert_disjoint, targets
from research.perturbations import perturb


def run(data_path: Path, run_path: Path, output: Path, settings: dict):
    data = load_dataset(data_path)
    summary = json.loads((run_path / "summary.json").read_text())
    config = AdapterConfig.model_validate(data.manifest["adapter"])
    if config.representation != "amplitude":
        raise ValueError("perturbations are defined for amplitude")
    y, labels = targets(data, summary["task"])
    conditions = settings["conditions"]
    rows, detail = [], []
    output.mkdir(parents=True, exist_ok=False)
    for split_path in sorted(run_path.glob("*/splits.csv")):
        with split_path.open(newline="") as handle:
            splits = list(csv.DictReader(handle))
        parts = {
            p: np.flatnonzero(
                np.isin(data.recordings, [r["recording"] for r in splits if r["partition"] == p])
            )
            for p in ("train", "validation", "test")
        }
        assert_disjoint(data, *parts.values())
        if any(not len(v) for v in parts.values()):
            raise ValueError("empty or unmatched split")
        train, test = parts["train"], parts["test"]
        training_hash = hashlib.sha256(
            "\n".join(sorted(set(data.recordings[train]))).encode()
        ).hexdigest()
        reference = fit_reference(data.x[train])
        save_model(
            output / split_path.parent.name / "normal-reference.csimodel",
            reference,
            config,
            data.manifest["dataset"],
            "tamper_robust_v1",
            training_details={
                "training_recordings_sha256": training_hash,
                "archive_sha256": data.manifest["archive_sha256"],
            },
        )
        for model_path in sorted(split_path.parent.glob("*.csimodel")):
            predictor = ModelPredictor.load(
                model_path, config, data.x[test[0]], config.sample_rate_hz
            )
            if (
                predictor.metadata.training_details.get("archive_sha256")
                != data.manifest["archive_sha256"]
            ):
                raise ValueError("model dataset provenance mismatch")
            if predictor.metadata.class_labels != labels:
                raise ValueError("model labels mismatch")
            provenance = predictor.metadata.training_details
            if (
                provenance.get("training_recordings_sha256") != training_hash
                or provenance.get("windowing") != data.manifest["windowing"]
                or provenance.get("outer_fold") != split_path.parent.name
            ):
                raise ValueError("model split or windowing provenance mismatch")
            for kind, level in conditions:
                changed = perturb(
                    data.x[test], kind, level, predictor.weights["scale"], settings["seed"]
                )
                predictions = np.array(
                    [
                        labels.index(predictor.predict(w, config.sample_rate_hz).label)
                        for w in changed
                    ]
                )
                metrics = classification_metrics(y[test], predictions, labels)
                flagged = float(
                    np.mean(anomaly_scores(changed, reference) > reference["threshold"][0])
                )
                row = {
                    "fold": split_path.parent.name,
                    "model": model_path.stem,
                    "condition": kind,
                    "level": level,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "anomaly_window_fraction": flagged,
                }
                rows.append(row)
                detail.append(row | {"metrics": metrics})
            print(f"completed {split_path.parent.name}/{model_path.stem}", flush=True)
    if not rows:
        raise ValueError("no completed models found")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": data.manifest["dataset"],
        "archive_sha256": data.manifest["archive_sha256"],
        "source_run": str(run_path),
        "settings": settings,
        "results": detail,
        "limitation": "Anomaly fraction is per-window exceedance, not sustained alert rate "
        "or malicious tamper detection accuracy.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=Path("configs/research.json"))
    args = parser.parse_args()
    run(args.data, args.run, args.output, json.loads(args.settings.read_text()))


if __name__ == "__main__":
    main()

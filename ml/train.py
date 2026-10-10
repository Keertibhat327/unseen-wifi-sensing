"""Reproducible grouped evaluations with saved split assignments, reports and safe models."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from detection.adapters import AdapterConfig
from detection.inference import ModelPredictor
from ml.artifacts import save_model
from ml.data import WindowDataset, load_dataset
from ml.metrics import classification_metrics
from ml.models import train_baseline, train_cnn
from ml.preprocessing import fit_standardizer, standardize
from ml.splits import Fold, make_folds, targets


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=42, ge=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    baseline_max_iter: int = Field(default=1000, gt=0)
    cnn_epochs: int = Field(default=10, gt=0)
    cnn_batch_size: int = Field(default=64, gt=0)
    cnn_learning_rate: float = Field(default=0.001, gt=0)
    cnn_channels: int = Field(default=16, ge=1, le=128)
    cnn_patience: int = Field(default=3, gt=0)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def recording_metrics(
    data: WindowDataset,
    indices: np.ndarray,
    y: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict:
    truth, predicted = [], []
    for recording in np.unique(data.recordings[indices]):
        mask = data.recordings[indices] == recording
        actual = np.unique(y[indices][mask])
        if len(actual) != 1:
            raise ValueError("one recording must not carry multiple targets")
        truth.append(actual[0])
        predicted.append(probabilities[mask].mean(axis=0).argmax())
    return classification_metrics(np.asarray(truth), np.asarray(predicted), labels)


def save_assignments(path: Path, data: WindowDataset, fold: Fold) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["partition", "recording", "person", "action", "windows"])
        for partition, indices in (
            ("train", fold.train),
            ("validation", fold.validation),
            ("test", fold.test),
        ):
            for recording in np.unique(data.recordings[indices]):
                selected = indices[data.recordings[indices] == recording]
                i = selected[0]
                writer.writerow(
                    [partition, recording, data.people[i], data.actions[i], len(selected)]
                )


def evaluate_fold(
    data: WindowDataset,
    fold: Fold,
    y: np.ndarray,
    labels: list[str],
    task: str,
    model_name: str,
    settings: TrainingSettings,
    output: Path,
) -> dict:
    config = AdapterConfig.model_validate(data.manifest["adapter"])
    scaler = fit_standardizer(data.x[fold.train])
    x = standardize(data.x[fold.train], scaler)
    validation_x = standardize(data.x[fold.validation], scaler)
    if model_name == "baseline":
        weights, training = train_baseline(
            x, y[fold.train], settings.seed, settings.baseline_max_iter
        )
        kind = "logistic_summary_v1"
    else:
        weights, training = train_cnn(
            x,
            y[fold.train],
            validation_x,
            y[fold.validation],
            len(labels),
            settings.seed,
            settings.cnn_epochs,
            settings.cnn_batch_size,
            settings.cnn_learning_rate,
            settings.cnn_channels,
            settings.cnn_patience,
        )
        kind = "tiny_cnn_v1"
    weights |= scaler
    # Model selection uses validation only. Test predictions happen AFTER the artifact is saved.
    model_path = output / f"{model_name}.csimodel"
    training_hash = hashlib.sha256(
        "\n".join(sorted(set(data.recordings[fold.train]))).encode()
    ).hexdigest()
    metadata = save_model(
        model_path,
        weights,
        config,
        f"RS2002/{data.manifest['dataset']}",
        kind,
        class_labels=labels,
        inference_config={"task": task, "channels": settings.cnn_channels},
        training_details={
            "outer_fold": fold.name,
            "seed": settings.seed,
            "archive_sha256": data.manifest["archive_sha256"],
            "training_recordings_sha256": training_hash,
            "windowing": data.manifest["windowing"],
            "selection": "fixed baseline or minimum inner-validation CNN loss",
            "settings": settings.model_dump(),
            "training": training,
        },
    )
    predictor = ModelPredictor.load(model_path, config, data.x[fold.test[0]], config.sample_rate_hz)
    probabilities = np.array(
        [
            list(predictor.predict(data.x[i], config.sample_rate_hz).probabilities.values())
            for i in fold.test
        ]
    )
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(y[fold.test], predictions, labels)
    report = {
        "model": model_name,
        "held_out": fold.name,
        "task": task,
        "train_windows": len(fold.train),
        "validation_windows": len(fold.validation),
        "test_windows": len(fold.test),
        "window_metrics": metrics,
        "recording_metrics": recording_metrics(data, fold.test, y, probabilities, labels),
        "model_metadata": metadata.model_dump(mode="json"),
        "test_label_scope": data.manifest["label_scope"],
    }
    write_json(output / f"{model_name}.metrics.json", report)
    with (output / f"{model_name}.predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window_index",
                "recording",
                "actual",
                "predicted",
                *[f"p_{label}" for label in labels],
            ]
        )
        for index, prediction, scores in zip(fold.test, predictions, probabilities, strict=True):
            writer.writerow(
                [
                    int(index),
                    data.recordings[index],
                    labels[y[index]],
                    labels[prediction],
                    *scores.tolist(),
                ]
            )
    print(
        f"{fold.name} / {model_name}: accuracy={metrics['accuracy']:.3f}, "
        f"macro-F1={metrics['macro_f1']:.3f}",
        flush=True,
    )
    return report


def run(
    data_path: Path,
    output: Path,
    task: str,
    protocol: str,
    models: list[str],
    settings: TrainingSettings,
    only_folds: list[str] | None = None,
) -> dict:
    if not models or len(set(models)) != len(models) or set(models) - {"baseline", "cnn"}:
        raise ValueError("choose unique baseline/cnn models")
    data = load_dataset(data_path)
    y, labels = targets(data, task)
    folds = make_folds(data, task, protocol, settings.seed, settings.validation_fraction)
    if only_folds:
        if set(only_folds) - {f.name for f in folds}:
            raise ValueError("unknown requested holdout")
        folds = [f for f in folds if f.name in only_folds]
    output.mkdir(parents=True, exist_ok=False)
    reports, skipped = [], []
    write_json(output / "dataset-audit.json", data.manifest)
    with threadpool_limits(limits=2):
        for fold in folds:
            folder = output / fold.name.replace("/", "_")
            folder.mkdir()
            save_assignments(folder / "splits.csv", data, fold)
            if fold.skip_reason:
                skipped.append({"fold": fold.name, "reason": fold.skip_reason})
                continue
            for model_name in models:
                reports.append(
                    evaluate_fold(data, fold, y, labels, task, model_name, settings, folder)
                )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "task": task,
        "protocol": protocol,
        "dataset": data.manifest["dataset"],
        "settings": settings.model_dump(),
        "partial_run": bool(only_folds),
        "skipped_folds": skipped,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "torch")},
        "models": {},
    }
    for name in models:
        selected = [r for r in reports if r["model"] == name]
        if not selected:
            continue
        summary["models"][name] = {
            "evaluated_folds": len(selected),
            "mean_fold_accuracy": float(
                np.mean([r["window_metrics"]["accuracy"] for r in selected])
            ),
            "mean_fold_macro_f1": float(
                np.mean([r["window_metrics"]["macro_f1"] for r in selected])
            ),
            "fold_macro_f1_std": float(np.std([r["window_metrics"]["macro_f1"] for r in selected])),
            "pooled_confusion_matrix": np.sum(
                [r["window_metrics"]["confusion_matrix"] for r in selected], axis=0
            ).tolist(),
            "labels": labels,
        }
    write_json(output / "summary.json", summary)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "held_out",
                "accuracy",
                "macro_f1",
                "recording_accuracy",
                "test_windows",
                "false_positive_rate",
                "fall_recall",
            ]
        )
        for report in reports:
            metrics = report["window_metrics"]
            writer.writerow(
                [
                    report["model"],
                    report["held_out"],
                    metrics["accuracy"],
                    metrics["macro_f1"],
                    report["recording_metrics"]["accuracy"],
                    report["test_windows"],
                    metrics.get("false_positive_rate"),
                    metrics.get("fall_recall"),
                ]
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing runs are never overwritten",
    )
    parser.add_argument("--task", choices=["action", "fall"], default="action")
    parser.add_argument("--protocol", choices=["lopo", "loao"], default="lopo")
    parser.add_argument(
        "--models", nargs="+", choices=["baseline", "cnn"], default=["baseline", "cnn"]
    )
    parser.add_argument("--only-fold", action="append")
    parser.add_argument("--settings", type=Path, default=Path("configs/ml.json"))
    args = parser.parse_args()
    summary = run(
        args.data,
        args.output,
        args.task,
        args.protocol,
        args.models,
        TrainingSettings.model_validate_json(args.settings.read_text()),
        args.only_fold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Offline known-participant leakage under held-out actions. Exports stay local."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from detection.adapters import AdapterConfig
from ml.artifacts import save_model
from ml.data import WindowDataset, load_dataset
from ml.metrics import classification_metrics
from ml.splits import assert_disjoint
from research.representations import represent


def privacy_folds(data: WindowDataset):
    for action in sorted(set(data.actions)):
        train = np.flatnonzero(data.actions != action)
        test = np.flatnonzero(data.actions == action)
        if set(data.people[train]) != set(data.people[test]):
            raise ValueError("each held-out action must cover all enrolled dataset participants")
        assert_disjoint(data, train, test)
        yield str(action), train, test


def run(data: WindowDataset, output: Path, factor: int = 4, alpha: float = 10.0) -> dict:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive")
    folds = list(privacy_folds(data))
    labels = sorted(set(data.people.tolist()))
    config = AdapterConfig.model_validate(data.manifest["adapter"])
    if len(labels) < 2:
        raise ValueError("privacy classification needs multiple participants")
    y = np.array([labels.index(p) for p in data.people])
    views = {kind: represent(data.x, kind, factor) for kind in ("raw", "reduced", "lower_rate")}
    output.mkdir(parents=True, exist_ok=False)
    results = []
    split_rows = []
    with threadpool_limits(limits=2):
        for action, train, test in folds:
            for partition, indices in (("train", train), ("test", test)):
                split_rows.extend(
                    {"fold": action, "partition": partition, "recording": str(r)}
                    for r in sorted(set(data.recordings[indices]))
                )
            majority = np.bincount(y[train]).argmax()
            for kind, x in views.items():
                model = make_pipeline(
                    StandardScaler(), RidgeClassifier(alpha=alpha, class_weight="balanced")
                )
                model.fit(x[train], y[train])
                scaler, classifier = model.steps[0][1], model.steps[1][1]
                save_model(
                    output / action / f"{kind}.csimodel",
                    {
                        "mean": scaler.mean_,
                        "scale": scaler.scale_,
                        "coef": classifier.coef_,
                        "intercept": classifier.intercept_,
                    },
                    config,
                    data.manifest["dataset"],
                    "privacy_ridge_v1",
                    labels,
                    {"task": "privacy_identity", "representation": kind, "factor": factor},
                    {
                        "held_out_action": action,
                        "alpha": alpha,
                        "archive_sha256": data.manifest["archive_sha256"],
                        "windowing": data.manifest["windowing"],
                        "local_research_only": True,
                    },
                )
                metrics = classification_metrics(y[test], model.predict(x[test]), labels)
                results.append({"action": action, "representation": kind, **metrics})
                print(f"{action} {kind}: accuracy={metrics['accuracy']:.3f}", flush=True)
            results.append(
                {
                    "action": action,
                    "representation": "majority_control",
                    **classification_metrics(y[test], np.full(len(test), majority), labels),
                }
            )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": data.manifest["dataset"],
        "archive_sha256": data.manifest["archive_sha256"],
        "protocol": "known participants, leave one action out; recording/content disjoint",
        "attacker": "train-only StandardScaler + balanced RidgeClassifier",
        "alpha": alpha,
        "lower_rate_factor": factor,
        "source_sample_rate_hz": data.manifest["adapter"]["sample_rate_hz"],
        "uniform_chance": 1 / len(labels),
        "participants": labels,
        "representations": {
            kind: {
                "mean_accuracy": float(
                    np.mean([r["accuracy"] for r in results if r["representation"] == kind])
                ),
                "mean_macro_f1": float(
                    np.mean([r["macro_f1"] for r in results if r["representation"] == kind])
                ),
            }
            for kind in (*views, "majority_control")
        },
        "folds": results,
        "limitation": "Dataset identity correlates with session/environment; "
        "not biometric validity or privacy guarantee. Attacker exports stay in local artifacts.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "partition", "recording"])
        writer.writeheader()
        writer.writerows(split_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--acknowledge-research-use", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_research_use:
        parser.error("explicit --acknowledge-research-use required for dataset identity evaluation")
    run(load_dataset(args.data), args.output, args.factor, args.alpha)


if __name__ == "__main__":
    main()

"""Publish aggregate offline research results; never copy identity templates or models."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def report(privacy: Path, robustness: Path, output: Path):
    p = json.loads((privacy / "summary.json").read_text())
    r = json.loads((robustness / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in r["results"]:
        groups[row["model"], row["condition"], row["level"]].append(row)
    aggregate = []
    for (model, kind, level), rows in groups.items():
        aggregate.append(
            {
                "model": model,
                "condition": kind,
                "level": level,
                "folds": len(rows),
                **{
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in ("accuracy", "macro_f1", "anomaly_window_fraction")
                },
            }
        )
    text = [
        "# M4 measured results",
        "",
        "Local WiFall experiments, 2026-10-10. All means weight each held-out fold equally.",
        "",
        "## Privacy leakage",
        "",
        "Known dataset participants, held-out actions; five folds. "
        "Train-only scaling and balanced ridge classifiers; no recording crosses partitions.",
        "",
        "| Representation | Mean accuracy | Mean macro-F1 |",
        "|---|---:|---:|",
    ]
    for kind, metrics in p["representations"].items():
        text.append(f"| {kind} | {metrics['mean_accuracy']:.1%} | {metrics['mean_macro_f1']:.3f} |")
    text += [
        "",
        "Uniform chance is 10%, but the majority control reaches 67.5% accuracy "
        "because ID2 dominates the data. Macro-F1 reveals minority-class behavior. "
        "Reduced features retain identity-related information and score higher macro-F1 "
        "than raw windows here. Lower sampling rate is not a privacy guarantee. "
        "Participant labels may correlate with collection session or environment; "
        "these results do not validate biometric identification.",
        "",
        "## Robustness",
        "",
        "Fixed M3 binary-fall models, ten held-out participants, seven conditions. "
        "Perturbations are software-only. No test data is used to retrain models or "
        "fit the statistical anomaly reference.",
        "",
        "| Model | Condition | Level | Accuracy | Macro-F1 | Flagged windows |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        text.append(
            f"| {row['model']} | {row['condition']} | {row['level']:g} | "
            f"{row['accuracy']:.1%} | {row['macro_f1']:.3f} | "
            f"{row['anomaly_window_fraction']:.1%} |"
        )
    text += [
        "",
        "Noise is measured in training carrier standard deviations; drift ends at "
        "the stated multiple of that scale. Packet loss is a drop probability with "
        "last-frame hold; the first frame is retained. Perturbations can sometimes improve "
        "a weak model's score, so the table reports changes without assuming "
        "monotonic degradation.",
        "",
        "Flagged windows are threshold exceedances, not gateway alert counts. "
        "Clean held-out windows can also be flagged. Anomalies indicate distribution "
        "changes, not intent, physical interference, or proven tampering. Gateway alerts "
        "require sustained evidence and are tested separately using synthetic signals.",
        "",
        "![Aggregate experiment chart](summary.png)",
        "",
        "Local provenance: `artifacts/m4/privacy-wifall-exported/` and "
        "`artifacts/m4/robustness-wifall-final/`. Research model weights and person-level "
        "predictions are not included in this aggregate report.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    (output / "privacy.json").write_text(json.dumps(p["representations"], indent=2))
    with (output / "robustness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(p["representations"])
    axes[0].bar(
        range(len(names)),
        [p["representations"][k]["mean_macro_f1"] for k in names],
        color="#237c83",
    )
    axes[0].set_xticks(range(len(names)), ["Raw", "Reduced", "Lower rate", "Majority"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean macro-F1")
    axes[0].set_title("Identity leakage: held-out actions")
    conditions = r["settings"]["conditions"]
    for model in sorted({row["model"] for row in aggregate}):
        values = [
            next(
                row["accuracy"]
                for row in aggregate
                if (row["model"], row["condition"], row["level"]) == (model, kind, level)
            )
            for kind, level in conditions
        ]
        axes[1].plot(range(len(values)), values, marker="o", label=model)
    axes[1].set_xticks(range(len(conditions)), [f"{k}\n{v:g}" for k, v in conditions], fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean accuracy")
    axes[1].set_title("Binary fall task: held-out participants")
    axes[1].legend()
    fig.suptitle("WiFall offline research — no hardware accuracy claim")
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privacy", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/m4"))
    args = parser.parse_args()
    report(args.privacy, args.robustness, args.output)


if __name__ == "__main__":
    main()

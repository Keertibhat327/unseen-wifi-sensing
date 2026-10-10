"""Collect completed experiments into a compact report without copying raw recordings or models."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def report(runs: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    text = [
        "# M3 measured results",
        "",
        "Completed local experiments; no hardware accuracy claim.",
        "",
        "Accuracy and macro-F1 below are **unweighted means across held-out folds**, "
        "not pooled-window scores.",
        "",
        "| Dataset / task / protocol | Model | Folds | Mean accuracy | Mean macro-F1 | F1 std |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rows = []
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        folder = output / run.name
        folder.mkdir(exist_ok=True)
        for name in ("summary.json", "results.csv"):
            shutil.copyfile(run / name, folder / name)
        for model, metrics in summary["models"].items():
            descriptor = f"{summary['dataset']} / {summary['task']} / {summary['protocol']}"
            text.append(
                f"| {descriptor} | {model} | {metrics['evaluated_folds']} | "
                f"{metrics['mean_fold_accuracy']:.1%} | {metrics['mean_fold_macro_f1']:.3f} | "
                f"{metrics['fold_macro_f1_std']:.3f} |"
            )
            rows.append(
                {
                    "experiment": run.name,
                    "model": model,
                    "folds": metrics["evaluated_folds"],
                    "accuracy": metrics["mean_fold_accuracy"],
                    "macro_f1": metrics["mean_fold_macro_f1"],
                    "f1_std": metrics["fold_macro_f1_std"],
                }
            )
        if summary["protocol"] == "lopo" and summary["task"] == "fall":
            fig, axes = plt.subplots(1, len(summary["models"]), figsize=(9, 4), squeeze=False)
            for ax, (model, metrics) in zip(axes[0], summary["models"].items(), strict=True):
                counts = np.asarray(metrics["pooled_confusion_matrix"])
                normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
                ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
                for i in range(len(counts)):
                    for j in range(len(counts)):
                        ax.text(
                            j,
                            i,
                            f"{counts[i, j]}\n({normalized[i, j]:.1%})",
                            ha="center",
                            va="center",
                            color="white" if normalized[i, j] > 0.55 else "black",
                        )
                ax.set_xticks(range(len(counts)), metrics["labels"])
                ax.set_yticks(range(len(counts)), metrics["labels"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual recording label")
                ax.set_title(model)
            fig.suptitle("WiFall binary task: pooled held-out windows")
            fig.text(
                0.5,
                0.015,
                "Counts are dominated by ID2; see fold means. Not calibrated alarm performance.",
                ha="center",
                fontsize=8,
            )
            fig.tight_layout(rect=(0, 0.06, 1, 0.94))
            fig.savefig(folder / "confusion.png", dpi=160)
            plt.close(fig)
    text += [
        "",
        "## Interpretation",
        "",
        "- These are untuned reference models. No test fold was used for scaling, "
        "early stopping or model selection.",
        "- WiFall ID2 contributes 1,406 of 2,076 windows; pooled metrics overrepresent ID2.",
        "- **LOAO is a negative-only false-alarm transfer diagnostic.** Its accuracy is "
        "specificity on an unseen non-fall action, not fall detection accuracy. "
        "Fall recall is null; held-out-fall is skipped because training has no positive class.",
        "- Recording labels are copied to windows; precise fall onset is not annotated. "
        "Scores therefore do not measure real-time alarm performance.",
        "- WiGesture excludes `dataset/dynamic/ID8/waveright.csv`: its rows say `waveleft`. "
        "Static action labels come from verified filenames, not participant codes in `taget`.",
        "- Models, split assignments, predictions and per-class reports remain in `artifacts/m3/`. "
        "Raw data and weights are not included in this report snapshot.",
        "- Models remain advisory. Validate on actual hardware and rooms before using decisions.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["experiment", "model", "folds", "accuracy", "macro_f1", "f1_std"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/m3"))
    args = parser.parse_args()
    report(args.runs, args.output)


if __name__ == "__main__":
    main()

"""Report supported and unsupported classes explicitly, including negative-only LOAO tests."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y: np.ndarray, predictions: np.ndarray, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predictions,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    result = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1.mean()),
        "macro_f1_definition": "all target classes, zero for unsupported classes",
        "classes_present": [labels[i] for i in range(len(labels)) if support[i]],
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=np.arange(len(labels))
        ).tolist(),
        "confusion_labels": labels,
    }
    if labels == ["nonfall", "fall"]:
        negatives, positives = y == 0, y == 1
        result["false_positive_rate"] = (
            float((predictions[negatives] == 1).mean()) if negatives.any() else None
        )
        result["fall_recall"] = (
            float((predictions[positives] == 1).mean()) if positives.any() else None
        )
    return result

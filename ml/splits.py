"""Outer domain holdouts, inner recording holdouts, and explicit leakage checks."""

from dataclasses import dataclass

import numpy as np

from ml.data import WindowDataset


@dataclass(frozen=True)
class Fold:
    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    skip_reason: str | None = None


def targets(data: WindowDataset, task: str) -> tuple[np.ndarray, list[str]]:
    if task == "fall":
        if "fall" not in data.actions:
            raise ValueError("fall task requires fall recordings")
        return (data.actions == "fall").astype(int), ["nonfall", "fall"]
    if task != "action":
        raise ValueError("only action and binary fall tasks are supported; identity is excluded")
    labels = sorted(set(data.actions.tolist()))
    return np.array([labels.index(a) for a in data.actions]), labels


def assert_disjoint(data: WindowDataset, *parts: np.ndarray) -> None:
    for i, left in enumerate(parts):
        for right in parts[i + 1 :]:
            if np.intersect1d(left, right).size:
                raise ValueError("window leakage across splits")
            for values in (data.recordings, data.content_hashes):
                if set(values[left]) & set(values[right]):
                    raise ValueError("recording or duplicate-content leakage across splits")


def make_folds(
    data: WindowDataset, task: str, protocol: str, seed: int, validation_fraction: float = 0.2
) -> list[Fold]:
    y, labels = targets(data, task)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if protocol not in ("lopo", "loao"):
        raise ValueError("protocol must be lopo or loao")
    if protocol == "loao" and task != "fall":
        raise ValueError(
            "LOAO action classification has an unseen target class; use binary fall task"
        )
    domain = data.people if protocol == "lopo" else data.actions
    folds = []
    for held_out in sorted(set(domain.tolist())):
        test = np.flatnonzero(domain == held_out)
        candidates = np.flatnonzero(domain != held_out)
        if len(set(y[candidates])) != len(labels):
            folds.append(
                Fold(
                    str(held_out),
                    candidates,
                    np.array([], dtype=int),
                    test,
                    "training set lacks a target class (cannot learn held-out fall)",
                )
            )
            continue
        rng = np.random.default_rng(seed)
        val_groups = []
        # Each content group is kept whole, even if the archive repeats it under another filename.
        for label in range(len(labels)):
            groups = np.unique(data.content_hashes[candidates[y[candidates] == label]])
            if len(groups) < 2:
                raise ValueError(
                    "need two independent training recordings per class for validation"
                )
            rng.shuffle(groups)
            count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
            val_groups.extend(groups[:count])
        is_val = np.isin(data.content_hashes[candidates], val_groups)
        train, validation = candidates[~is_val], candidates[is_val]
        assert_disjoint(data, train, validation, test)
        if len(set(y[train])) != len(labels):
            raise ValueError("training split lost a class")
        folds.append(Fold(str(held_out), train, validation, test))
    return folds

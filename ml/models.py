"""Small, original reference models; no upstream research implementation is copied."""

import copy

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from ml.preprocessing import summary_features


class TinyCNN(nn.Module):
    def __init__(self, subcarriers: int, classes: int, channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(subcarriers, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1)
        self.head = nn.Linear(channels * 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Public input is batch × time × carriers; Conv1d uses batch × channels × time.
        x = torch.relu(self.conv1(x.transpose(1, 2)))
        x = torch.relu(self.conv2(x)).mean(dim=2)
        return self.head(x)


def train_baseline(
    x: np.ndarray, y: np.ndarray, seed: int, max_iter: int
) -> tuple[dict[str, np.ndarray], dict]:
    model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)
    model.fit(summary_features(x), y)
    if np.any(model.n_iter_ >= max_iter):
        raise ValueError("baseline did not converge; increase max_iter before reporting metrics")
    return {"coef": model.coef_, "intercept": model.intercept_}, {
        "iterations": model.n_iter_.tolist(),
        "training_accuracy": float(model.score(summary_features(x), y)),
    }


def train_cnn(
    x: np.ndarray,
    y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    channels: int,
    patience: int,
) -> tuple[dict[str, np.ndarray], dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = TinyCNN(x.shape[2], classes, channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    counts = np.bincount(y, minlength=classes)
    if np.any(counts == 0):
        raise ValueError("CNN training requires all classes")
    weights = torch.tensor(len(y) / (classes * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    train_x, train_y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
    val_x, val_y = torch.from_numpy(validation_x), torch.tensor(validation_y, dtype=torch.long)
    rng = np.random.default_rng(seed)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        total = 0.0
        for offset in range(0, len(x), batch_size):
            indices = order[offset : offset + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(val_x), val_y))
        if not np.isfinite(validation_loss):
            raise ValueError("non-finite validation loss")
        history.append(
            {"epoch": epoch, "train_loss": total / len(x), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss, best_state, best_epoch = (
                validation_loss,
                copy.deepcopy(model.state_dict()),
                epoch,
            )
        elif epoch - best_epoch >= patience:
            break
    assert best_state is not None
    return {name: value.numpy() for name, value in best_state.items()}, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }

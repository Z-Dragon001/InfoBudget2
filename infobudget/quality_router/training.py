"""Supervised Huber-loss training for scalar local Fact quality."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from infobudget.quality_router.model import CapabilityConditionedQualityScorer


@dataclass(frozen=True, slots=True)
class QualityTrainingConfig:
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    batch_size: int = 128
    epochs: int = 100
    early_stopping_patience: int = 10
    huber_delta: float = 0.1
    seed: int = 42


@dataclass(frozen=True, slots=True)
class QualityTrainingResult:
    best_epoch: int
    best_validation_mae: float
    history: tuple[dict[str, float], ...]


def train_quality_scorer(
    model: CapabilityConditionedQualityScorer,
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    config: QualityTrainingConfig,
    device: str | torch.device = "cpu",
) -> QualityTrainingResult:
    _validate_dataset(train_features, train_targets, "train")
    _validate_dataset(validation_features, validation_targets, "validation")
    if train_features.shape[1] != model.input_dimension or validation_features.shape[1] != model.input_dimension:
        raise ValueError("feature dimension does not match quality scorer")
    if config.batch_size <= 0 or config.epochs <= 0 or config.early_stopping_patience <= 0:
        raise ValueError("batch_size, epochs, and early_stopping_patience must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.huber_delta <= 0:
        raise ValueError("invalid optimizer or Huber configuration")

    _seed_everything(config.seed)
    target_device = torch.device(device)
    model.to(target_device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_features, dtype=torch.float32),
            torch.as_tensor(train_targets, dtype=torch.float32),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.HuberLoss(delta=config.huber_delta)
    best_state: dict | None = None
    best_epoch = 0
    best_mae = float("inf")
    patience = 0
    history: list[dict[str, float]] = []

    validation_tensor = torch.as_tensor(validation_features, dtype=torch.float32, device=target_device)
    validation_target_tensor = torch.as_tensor(validation_targets, dtype=torch.float32, device=target_device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(target_device)
            batch_targets = batch_targets.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_targets)
            seen += len(batch_targets)
        model.eval()
        with torch.inference_mode():
            validation_predictions = model(validation_tensor)
            validation_mae = float(torch.mean(torch.abs(validation_predictions - validation_target_tensor)))
        history.append(
            {
                "epoch": float(epoch),
                "train_huber": total_loss / seen,
                "validation_mae": validation_mae,
            }
        )
        if validation_mae < best_mae - 1e-8:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("quality scorer training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return QualityTrainingResult(best_epoch, best_mae, tuple(history))


def _validate_dataset(features: np.ndarray, targets: np.ndarray, name: str) -> None:
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError(f"invalid {name} feature/target shapes")
    if not len(targets):
        raise ValueError(f"{name} dataset is empty")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"{name} dataset contains non-finite values")
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError(f"{name} targets must be in [0, 1]")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

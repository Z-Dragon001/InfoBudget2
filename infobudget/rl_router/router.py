"""Leakage-safe router policies and baselines."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from infobudget.rl_router.embedding import Encoder
from infobudget.rl_router.schemas import TIERS, Tier, TopicSegment


@dataclass(slots=True)
class RouteOutput:
    tier: Tier
    action_index: int
    probability: float
    probabilities: list[float]
    value: float


@dataclass(slots=True)
class FeatureScaler:
    mean: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean.tolist(), scale.tolist())

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.scale, dtype=np.float32)


class SegmentFeatureBuilder:
    """Builds features from segment text/structure only; QA fields are not accepted."""

    numeric_dimension = 6

    def __init__(self, encoder: Encoder, scaler: FeatureScaler | None = None):
        self.encoder = encoder
        self.scaler = scaler

    def fit(self, segments: list[TopicSegment]) -> FeatureScaler:
        self.scaler = FeatureScaler.fit(self.numeric(segments))
        return self.scaler

    def build(self, segments: list[TopicSegment]) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("feature scaler must be fitted on the training split")
        embeddings = self.encoder.encode([segment.text for segment in segments])
        numeric = self.scaler.transform(self.numeric(segments))
        return np.concatenate([embeddings, numeric], axis=1).astype(np.float32)

    @staticmethod
    def numeric(segments: list[TopicSegment]) -> np.ndarray:
        rows = []
        for segment in segments:
            rows.append(
                [
                    float(segment.token_count),
                    float(len(segment.turn_ids)),
                    float(len(segment.text)),
                    float(segment.text.count("\n") + 1),
                    float(sum(char.isdigit() for char in segment.text)),
                    float(len(set(segment.text.lower().split()))),
                ]
            )
        return np.asarray(rows, dtype=np.float32)


class RouterPolicy(ABC):
    router_type: str

    @abstractmethod
    def route(self, features: np.ndarray, *, deterministic: bool = False) -> list[RouteOutput]: ...


class EmbeddingMLPRouter(nn.Module, RouterPolicy):
    router_type = "embedding_mlp"

    def __init__(self, input_dimension: int, hidden_dimensions: list[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dimension
        for hidden in hidden_dimensions:
            layers.extend([nn.Linear(previous, hidden), nn.GELU(), nn.Dropout(dropout)])
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(previous, len(TIERS))
        self.value_head = nn.Linear(previous, 1)
        self.input_dimension = input_dimension
        self.hidden_dimensions = list(hidden_dimensions)
        self.dropout = dropout

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def route(self, features: np.ndarray, *, deterministic: bool = False) -> list[RouteOutput]:
        self.eval()
        with torch.inference_mode():
            logits, values = self(torch.as_tensor(features, dtype=torch.float32, device=next(self.parameters()).device))
            probabilities = torch.softmax(logits, dim=-1)
            actions = probabilities.argmax(-1) if deterministic else torch.distributions.Categorical(probabilities).sample()
        return [
            RouteOutput(TIERS[int(action)], int(action), float(probs[int(action)]), probs.tolist(), float(value))
            for action, probs, value in zip(actions.cpu(), probabilities.cpu(), values.cpu())
        ]

    def save_checkpoint(self, path: str | Path, scaler: FeatureScaler, metadata: dict | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dimension": self.input_dimension,
                "hidden_dimensions": self.hidden_dimensions,
                "dropout": self.dropout,
                "feature_scaler": asdict(scaler),
                "router_type": self.router_type,
                "metadata": metadata or {},
            },
            target,
        )
        return target

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cpu") -> tuple["EmbeddingMLPRouter", FeatureScaler, dict]:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("router_type") != "embedding_mlp":
            raise ValueError("checkpoint is not an embedding_mlp router")
        model = cls(payload["input_dimension"], payload["hidden_dimensions"], payload["dropout"])
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        return model, FeatureScaler(**payload["feature_scaler"]), payload.get("metadata", {})


class ConstantRouter(RouterPolicy):
    router_type = "constant"

    def __init__(self, tier: Tier): self.tier = tier

    def route(self, features: np.ndarray, *, deterministic: bool = False) -> list[RouteOutput]:
        index = TIERS.index(self.tier)
        probabilities = [1.0 if i == index else 0.0 for i in range(3)]
        return [RouteOutput(self.tier, index, 1.0, probabilities, 0.0) for _ in features]


class RandomRouter(RouterPolicy):
    router_type = "random"

    def __init__(self, seed: int = 42): self.random = random.Random(seed)

    def route(self, features: np.ndarray, *, deterministic: bool = False) -> list[RouteOutput]:
        return [RouteOutput(tier := self.random.choice(TIERS), TIERS.index(tier), 1 / 3, [1 / 3] * 3, 0.0) for _ in features]


class LengthHeuristicRouter(RouterPolicy):
    router_type = "length_heuristic"

    def __init__(self, small_threshold: float = -0.5, large_threshold: float = 0.5):
        self.small_threshold, self.large_threshold = small_threshold, large_threshold

    def route(self, features: np.ndarray, *, deterministic: bool = False) -> list[RouteOutput]:
        # The first normalized structural feature is token_count.
        outputs = []
        for value in features[:, -SegmentFeatureBuilder.numeric_dimension]:
            tier: Tier = "small" if value < self.small_threshold else "large" if value > self.large_threshold else "medium"
            index = TIERS.index(tier)
            outputs.append(RouteOutput(tier, index, 1.0, [1.0 if i == index else 0.0 for i in range(3)], 0.0))
        return outputs

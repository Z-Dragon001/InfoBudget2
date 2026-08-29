"""Capability-conditioned scalar quality scorer and feature construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from infobudget.quality_router.schemas import (
    CAPABILITY_DIMENSIONS,
    ModelCapabilityProfile,
)
from infobudget.rl_router.embedding import Encoder
from infobudget.rl_router.router import FeatureScaler, SegmentFeatureBuilder
from infobudget.rl_router.schemas import TopicSegment


class QualityFeatureBuilder:
    """Concatenate segment embedding/structure with a frozen MemoryPrint."""

    capability_dimension = len(CAPABILITY_DIMENSIONS)

    def __init__(self, encoder: Encoder, scaler: FeatureScaler | None = None):
        self.segment_builder = SegmentFeatureBuilder(encoder, scaler)

    @property
    def encoder(self) -> Encoder:
        return self.segment_builder.encoder

    @property
    def scaler(self) -> FeatureScaler | None:
        return self.segment_builder.scaler

    @property
    def input_dimension(self) -> int:
        return (
            self.encoder.dimension
            + SegmentFeatureBuilder.numeric_dimension
            + self.capability_dimension
        )

    def fit(self, segments: list[TopicSegment]) -> FeatureScaler:
        return self.segment_builder.fit(segments)

    def build_segment_features(self, segments: list[TopicSegment]) -> np.ndarray:
        return self.segment_builder.build(segments)

    def combine(
        self,
        segment_features: np.ndarray,
        profiles: Iterable[ModelCapabilityProfile],
    ) -> np.ndarray:
        profile_list = list(profiles)
        if len(segment_features) != len(profile_list):
            raise ValueError("segment feature/profile row count mismatch")
        capabilities = np.asarray(
            [profile.vector() for profile in profile_list], dtype=np.float32
        )
        if capabilities.shape != (len(profile_list), self.capability_dimension):
            raise ValueError(f"unexpected capability shape: {capabilities.shape}")
        return np.concatenate([segment_features, capabilities], axis=1).astype(
            np.float32
        )


class CapabilityConditionedQualityScorer(nn.Module):
    router_type = "capability_conditioned_quality_v1"

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dimension <= 0 or not hidden_dimensions or any(value <= 0 for value in hidden_dimensions):
            raise ValueError("quality scorer dimensions must be positive")
        layers: list[nn.Module] = []
        previous = input_dimension
        for hidden in hidden_dimensions:
            layers.extend([nn.Linear(previous, hidden), nn.GELU(), nn.Dropout(dropout)])
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.quality_head = nn.Linear(previous, 1)
        self.input_dimension = int(input_dimension)
        self.hidden_dimensions = list(hidden_dimensions)
        self.dropout = float(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.quality_head(self.backbone(features))).squeeze(-1)

    def predict(self, features: np.ndarray, *, device: str | torch.device | None = None) -> np.ndarray:
        self.eval()
        target_device = torch.device(device) if device is not None else next(self.parameters()).device
        with torch.inference_mode():
            values = self(torch.as_tensor(features, dtype=torch.float32, device=target_device))
        return values.cpu().numpy().astype(np.float32)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        scaler: FeatureScaler,
        embedding_model: str,
        embedding_dimension: int,
        metadata: dict | None = None,
    ) -> Path:
        if self.input_dimension != embedding_dimension + SegmentFeatureBuilder.numeric_dimension + len(CAPABILITY_DIMENSIONS):
            raise ValueError("checkpoint embedding/input dimensions are inconsistent")
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
                "embedding_model": embedding_model,
                "embedding_dimension": int(embedding_dimension),
                "capability_dimensions": list(CAPABILITY_DIMENSIONS),
                "metadata": metadata or {},
            },
            target,
        )
        return target

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, device: str = "cpu"
    ) -> tuple["CapabilityConditionedQualityScorer", FeatureScaler, dict]:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("router_type") != cls.router_type:
            raise ValueError("checkpoint is not a capability-conditioned quality scorer")
        if tuple(payload.get("capability_dimensions", ())) != CAPABILITY_DIMENSIONS:
            raise ValueError("checkpoint MemoryPrint dimensions do not match the current schema")
        embedding_dimension = int(payload["embedding_dimension"])
        expected_input = embedding_dimension + SegmentFeatureBuilder.numeric_dimension + len(CAPABILITY_DIMENSIONS)
        if int(payload["input_dimension"]) != expected_input:
            raise ValueError("checkpoint input dimension is internally inconsistent")
        model = cls(
            int(payload["input_dimension"]),
            [int(value) for value in payload["hidden_dimensions"]],
            float(payload["dropout"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        metadata = {
            **payload.get("metadata", {}),
            "embedding_model": payload["embedding_model"],
            "embedding_dimension": embedding_dimension,
        }
        return model, FeatureScaler(**payload["feature_scaler"]), metadata

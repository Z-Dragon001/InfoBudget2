"""Load the preprocessing, segmentation, model, and price configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infobudget.schemas import ModelSpec, PriceSpec


@dataclass(slots=True)
class ProjectConfig:
    name: str
    seed: int
    output_dir: str


@dataclass(slots=True)
class DatasetConfig:
    root_dir: str
    raw_dir: str
    processed_dir: str
    supported_datasets: list[str]
    default_splits: list[str]
    fallback_split_name: str = "full"
    store_flat_questions: bool = True
    store_flat_sessions: bool = True
    store_flat_turns: bool = True
    schema_version: str = "processed_v3"


@dataclass(slots=True)
class SegmentationConfig:
    method: str
    preserve_session_boundaries: bool
    bert_model_dir: str
    bert_mlp_checkpoint: str
    bert_max_length: int
    bert_batch_size: int
    adaptive_alpha: float
    bert_mlp_activation: str
    min_boundary_gap: int
    min_segment_turns: int
    min_segment_tokens: int
    max_segment_turns: int
    max_segment_tokens: int
    merge_short_segment: bool
    trace_enabled: bool = True


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig
    dataset: DatasetConfig
    segmentation: SegmentationConfig


@dataclass(slots=True)
class ProjectBundle:
    root_dir: Path
    config: AppConfig
    models: dict[str, ModelSpec]
    prices: dict[str, PriceSpec]
    prompt_dir: Path


def load_project_bundle(config_dir: str | Path = "configs") -> ProjectBundle:
    config_path = Path(config_dir)
    raw = _read_yaml(config_path / "config.yaml")
    app = AppConfig(
        project=ProjectConfig(**raw["project"]),
        dataset=DatasetConfig(**raw["dataset"]),
        segmentation=SegmentationConfig(**raw["segmentation"]),
    )
    models = {
        name: ModelSpec(**item)
        for name, item in _read_yaml(config_path / "models.yaml")["models"].items()
    }
    prices = {
        name: PriceSpec(**item)
        for name, item in _read_yaml(config_path / "prices.yaml")["prices"].items()
    }
    return ProjectBundle(
        root_dir=config_path.parent.resolve(),
        config=app,
        models=models,
        prices=prices,
        prompt_dir=config_path / "prompts",
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return data

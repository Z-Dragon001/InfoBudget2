"""Load the preprocessing, segmentation, model, and price configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infobudget.schemas import ModelSpec, PriceSpec


ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
    root_dir = config_path.parent.resolve()
    load_env_file(root_dir / ".env")
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
        root_dir=root_dir,
        config=app,
        models=models,
        prices=prices,
        prompt_dir=config_path / "prompts",
    )


def load_env_file(path: str | Path) -> bool:
    """Load a small project-local .env file without overriding process variables."""

    source = Path(path)
    if not source.is_file():
        return False
    for line_number, original in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid .env entry at {source}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
    return True


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return data

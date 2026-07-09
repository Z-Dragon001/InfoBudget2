"""功能：加载 YAML 配置与注册信息。
输入：配置目录路径。
输出：项目配置 bundle。
依赖：pathlib、dataclasses、yaml。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infobudget.schemas import ModelSpec, PriceSpec


@dataclass(slots=True)
class ProjectConfig:
    """项目级配置。"""

    name: str
    seed: int
    log_dir: str
    output_dir: str


@dataclass(slots=True)
class DatasetConfig:
    """数据集存储与预处理配置。"""

    root_dir: str
    raw_dir: str
    processed_dir: str
    supported_datasets: list[str]
    default_splits: list[str]
    fallback_split_name: str = "full"
    store_flat_questions: bool = True
    store_flat_sessions: bool = True


@dataclass(slots=True)
class SegmentationConfig:
    """分段配置。"""

    method: str
    embedding_model: str
    similarity_threshold: float
    drop_threshold: float
    enable_smoothing: bool
    smooth_alpha: float
    min_boundary_gap: int
    min_segment_turns: int
    min_segment_tokens: int
    max_segment_turns: int
    max_segment_tokens: int
    merge_short_segment: bool
    trace_enabled: bool = True


@dataclass(slots=True)
class ScoringConfig:
    """评分主配置。"""

    tokenizer_name: str
    novelty_top_k: int
    semantic_embedding_model: str
    episodic_embedding_model: str
    spacy_model: str


@dataclass(slots=True)
class RouterConfig:
    """路由配置。"""

    method: str
    p33: float
    p67: float
    deferred_fit: bool
    trace_enabled: bool = True


@dataclass(slots=True)
class ExtractorConfig:
    """提取器配置。"""

    mode: str
    max_new_tokens: int
    json_mode: bool
    local_backend: str
    api_backend: str
    fallback_to_mock: bool = True
    extraction_mode: str = "flat"


@dataclass(slots=True)
class StorageConfig:
    """存储配置。"""

    jsonl_dir: str
    qdrant_dir: str
    normalize_embeddings: bool
    qdrant_memory_collection: str
    qdrant_episode_collection: str


@dataclass(slots=True)
class EvaluationConfig:
    """评估配置。"""

    datasets: list[str]
    judge_mode: str
    track_build_stage: bool
    track_qa_stage: bool
    save_predictions: bool
    qa_mode: str = "llm_qa"
    answer_model_tier: str = "medium"
    qa_max_new_tokens: int = 512
    judge_model: ModelSpec | None = None
    retrieval_top_k: int = 5
    locomo_retrieval_top_k: int = 60
    longmemeval_retrieval_top_k: int = 20
    save_retrieval_traces: bool = True


@dataclass(slots=True)
class IntrinsicWeights:
    """内在信息指标权重。"""

    entropy: float
    lexical_density: float
    entity_density: float
    concept_density: float


@dataclass(slots=True)
class UtilityWeights:
    """效用指标权重。"""

    information_gain: float
    actionability: float


@dataclass(slots=True)
class FusionWeights:
    """融合权重。"""

    intrinsic_weight: float
    utility_weight: float


@dataclass(slots=True)
class WeightConfig:
    """评分权重配置。"""

    intrinsic: IntrinsicWeights
    utility: UtilityWeights
    fusion: FusionWeights


@dataclass(slots=True)
class AppConfig:
    """应用配置。"""

    project: ProjectConfig
    dataset: DatasetConfig
    segmentation: SegmentationConfig
    scoring: ScoringConfig
    router: RouterConfig
    extractor: ExtractorConfig
    storage: StorageConfig
    evaluation: EvaluationConfig


@dataclass(slots=True)
class ProjectBundle:
    """完整配置 bundle。"""

    root_dir: Path
    config: AppConfig
    weights: WeightConfig
    models: dict[str, ModelSpec]
    prices: dict[str, PriceSpec]
    prompt_dir: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是对象结构: {path}")
    return data


def _build_app_config(raw: dict[str, Any]) -> AppConfig:
    """构建主配置。"""
    return AppConfig(
        project=ProjectConfig(**raw["project"]),
        dataset=DatasetConfig(**raw["dataset"]),
        segmentation=SegmentationConfig(**raw["segmentation"]),
        scoring=ScoringConfig(**raw["scoring"]),
        router=RouterConfig(**raw["router"]),
        extractor=ExtractorConfig(**raw["extractor"]),
        storage=StorageConfig(**raw["storage"]),
        evaluation=_build_evaluation_config(raw),
    )


def _build_evaluation_config(raw: dict[str, Any]) -> EvaluationConfig:
    """Build evaluation config and hydrate the optional judge model spec."""
    payload = dict(raw["evaluation"])
    judge_model_raw = payload.pop("judge_model", None)
    judge_model = ModelSpec(**judge_model_raw) if isinstance(judge_model_raw, dict) else None
    return EvaluationConfig(**payload, judge_model=judge_model)


def _build_weights(raw: dict[str, Any]) -> WeightConfig:
    """构建权重配置。"""
    section = raw["scoring"]
    return WeightConfig(
        intrinsic=IntrinsicWeights(**section["intrinsic"]),
        utility=UtilityWeights(**section["utility"]),
        fusion=FusionWeights(**section["fusion"]),
    )


def _build_models(raw: dict[str, Any]) -> dict[str, ModelSpec]:
    """构建模型注册表。"""
    return {name: ModelSpec(**item) for name, item in raw["models"].items()}


def _build_prices(raw: dict[str, Any]) -> dict[str, PriceSpec]:
    """构建价格注册表。"""
    return {name: PriceSpec(**item) for name, item in raw["prices"].items()}


def load_project_bundle(config_dir: str | Path = "configs") -> ProjectBundle:
    """加载完整项目配置。"""
    config_path = Path(config_dir)
    root_dir = config_path.parent.resolve()
    app = _build_app_config(_read_yaml(config_path / "config.yaml"))
    weights = _build_weights(_read_yaml(config_path / "weights.yaml"))
    models = _build_models(_read_yaml(config_path / "models.yaml"))
    prices = _build_prices(_read_yaml(config_path / "prices.yaml"))
    return ProjectBundle(
        root_dir=root_dir,
        config=app,
        weights=weights,
        models=models,
        prices=prices,
        prompt_dir=config_path / "prompts",
    )

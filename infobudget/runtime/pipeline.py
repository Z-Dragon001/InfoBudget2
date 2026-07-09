"""功能：串联 InfoBudget 第一阶段端到端流程。
输入：Turn 列表与配置目录。
输出：分段、评分、路由、记忆、成本与评估结果。
依赖：配置、分段、评分、提取、存储、评估。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from infobudget.config import ProjectBundle, load_project_bundle
from infobudget.cost.logger import CostLogger
from infobudget.evaluation.metrics import EvaluationMetrics, aggregate_metrics
from infobudget.extractors.llm_joint import APIJointExtractor, LocalJointExtractor, TieredJointExtractor
from infobudget.extractors.mock_joint import MockJointExtractor
from infobudget.memory.store import MemoryStore
from infobudget.routing.fixed_percentile import BudgetAwareRouter
from infobudget.runtime.model_registry import ModelRegistry, PriceRegistry
from infobudget.runtime.prompt_loader import load_prompt_map
from infobudget.schemas import MemoryEntry, ScoreResult, Segment, Turn
from infobudget.scoring.modes import ScoringMode, normalize_scoring_mode
from infobudget.scoring.scorer import InformationScorer
from infobudget.segmentation.lite_topic_seg import LiteTopicSeg
from infobudget.utils.embeddings import build_text_encoder
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """端到端运行结果。"""

    segments: list[Segment]
    scores: list[ScoreResult]
    tiers: list[str]
    entries: list[MemoryEntry]
    metrics: EvaluationMetrics


class InfoBudgetPipeline:
    """InfoBudget 第一阶段默认流水线。"""

    def __init__(
        self,
        bundle: ProjectBundle,
        scoring_mode: ScoringMode = "full",
        run_output_dir: str | Path | None = None,
    ):
        self.bundle = bundle
        self.scoring_mode = normalize_scoring_mode(scoring_mode)
        self.run_output_dir = Path(run_output_dir).resolve() if run_output_dir else None
        self.encoder = build_text_encoder(bundle.config.segmentation.embedding_model)
        self.segmenter = LiteTopicSeg(bundle.config.segmentation, self.encoder)
        self.scorer = InformationScorer(bundle.config.scoring, bundle.weights, self.encoder, self.scoring_mode)
        self.router = BudgetAwareRouter(bundle.config.router.p33, bundle.config.router.p67)
        self.model_registry = ModelRegistry(bundle.models)
        self.price_registry = PriceRegistry(bundle.prices)
        storage_cfg = bundle.config.storage
        storage_root = bundle.root_dir
        if self.run_output_dir:
            storage_cfg = replace(
                storage_cfg,
                jsonl_dir="memory_jsonl",
                qdrant_dir="qdrant",
            )
            storage_root = self.run_output_dir
        cost_path = storage_root / storage_cfg.jsonl_dir / "cost_logs.jsonl"
        self.cost_logger = CostLogger(self.price_registry, cost_path)
        self.memory_store = MemoryStore(storage_cfg, storage_root)
        prompts = load_prompt_map(
            bundle.prompt_dir,
            {
                "small": "joint_memory_extraction_small.txt",
                "medium": "joint_memory_extraction_medium.txt",
                "large": "joint_memory_extraction_large.txt",
                "default": "joint_memory_extraction.txt",
            },
        )
        relational_prompts = load_prompt_map(
            bundle.prompt_dir,
            {
                "small": "joint_memory_relation_small.txt",
                "medium": "joint_memory_relation_medium.txt",
                "large": "joint_memory_relation_large.txt",
                "default": "joint_memory_relation.txt",
            },
        )
        extraction_mode = bundle.config.extractor.extraction_mode
        mock_extractor = MockJointExtractor(self.model_registry, self.cost_logger, prompts, extraction_mode=extraction_mode)
        if bundle.config.extractor.mode == "mock_joint":
            self.extractor = mock_extractor
        else:
            self.extractor = TieredJointExtractor(
                model_registry=self.model_registry,
                local_extractor=LocalJointExtractor(
                    self.model_registry,
                    self.cost_logger,
                    prompts,
                    relational_prompt_template=relational_prompts,
                    max_new_tokens=bundle.config.extractor.max_new_tokens,
                    json_mode=bundle.config.extractor.json_mode,
                    extractor_name="local_joint_extractor",
                    extraction_mode=extraction_mode,
                ),
                api_extractor=APIJointExtractor(
                    self.model_registry,
                    self.cost_logger,
                    prompts,
                    relational_prompt_template=relational_prompts,
                    max_new_tokens=bundle.config.extractor.max_new_tokens,
                    json_mode=bundle.config.extractor.json_mode,
                    extractor_name="api_joint_extractor",
                    extraction_mode=extraction_mode,
                ),
                fallback_extractor=mock_extractor,
                fallback_on_error=bundle.config.extractor.fallback_to_mock,
            )

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str | Path = "configs",
        scoring_mode: ScoringMode = "full",
        run_output_dir: str | Path | None = None,
    ) -> "InfoBudgetPipeline":
        return cls(load_project_bundle(config_dir), scoring_mode, run_output_dir)

    def process_turns(self, turns: list[Turn], save_outputs: bool = True) -> PipelineResult:
        """运行完整构建流程。"""
        segments = self.segmenter.segment(turns)
        self.memory_store.record_segments(segments)
        scores: list[ScoreResult] = []
        tiers: list[str] = []
        entries: list[MemoryEntry] = []
        for segment in segments:
            score = self.scorer.score(segment, self.memory_store)
            tier = self.router.route(score.final_score)
            extracted_entries = self.extractor.extract(segment, tier, score)
            for entry in extracted_entries:
                memory_embedding = self.encoder.encode_text(entry.memory)
                self.memory_store.add_entry(entry, memory_embedding)
                entries.append(entry)
            scores.append(score)
            tiers.append(tier)
        metrics = aggregate_metrics(
            correctness=[True for _ in entries],
            cost_logs=self.cost_logger.logs,
            routed_tiers=tiers,
            dataset_name="pipeline_debug",
            split="adhoc",
            num_examples=1,
            num_queries=max(1, len(turns)),
            num_memories=len(entries),
            qa_latency_ms=0,
        )
        if save_outputs:
            self._save_outputs(metrics)
        logger.info("Pipeline completed with %s memories", len(entries))
        return PipelineResult(segments=segments, scores=scores, tiers=tiers, entries=entries, metrics=metrics)

    def _save_outputs(self, metrics: EvaluationMetrics) -> None:
        self.save_memory_outputs()
        output_dir = self.bundle.root_dir / self.bundle.config.project.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics.to_dict(), handle, ensure_ascii=False, indent=2)
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for entry in self.memory_store.entries:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        with (output_dir / "run_summary.md").open("w", encoding="utf-8") as handle:
            handle.write("# InfoBudget Run Summary\n\n")
            handle.write(f"- memories: {len(self.memory_store.entries)}\n")
            handle.write(f"- total_cost_usd: {metrics.total_cost_usd}\n")
            handle.write(f"- accuracy: {metrics.accuracy}\n")

    def save_memory_outputs(self) -> None:
        """Persist memory JSONL, Qdrant collections, segments, and cost logs."""
        self.memory_store.save()
        self.cost_logger.save()

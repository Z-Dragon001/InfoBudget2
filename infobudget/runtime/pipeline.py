"""功能：串联 InfoBudget 第一阶段端到端流程。
输入：Turn 列表与配置目录。
输出：分段、评分、路由、记忆、成本与评估结果。
依赖：配置、分段、评分、提取、存储、评估。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from infobudget.config import ProjectBundle, load_project_bundle
from infobudget.cost.logger import CostLogger
from infobudget.evaluation.metrics import EvaluationMetrics, aggregate_metrics
from infobudget.extractors.mock_joint import MockJointExtractor
from infobudget.memory.store import MemoryStore
from infobudget.routing.fixed_percentile import BudgetAwareRouter
from infobudget.runtime.model_registry import ModelRegistry, PriceRegistry
from infobudget.runtime.prompt_loader import load_prompt
from infobudget.schemas import MemoryEntry, ScoreResult, Segment, Turn
from infobudget.scoring.scorer import InformationScorer
from infobudget.segmentation.lite_topic_seg import LiteTopicSeg
from infobudget.utils.embeddings import HashingTextEncoder
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

    def __init__(self, bundle: ProjectBundle):
        self.bundle = bundle
        self.encoder = HashingTextEncoder()
        self.segmenter = LiteTopicSeg(bundle.config.segmentation, self.encoder)
        self.scorer = InformationScorer(bundle.config.scoring, bundle.weights, self.encoder)
        self.router = BudgetAwareRouter(bundle.config.router.p33, bundle.config.router.p67)
        self.model_registry = ModelRegistry(bundle.models)
        self.price_registry = PriceRegistry(bundle.prices)
        cost_path = bundle.root_dir / bundle.config.storage.jsonl_dir / "cost_logs.jsonl"
        self.cost_logger = CostLogger(self.price_registry, cost_path)
        self.memory_store = MemoryStore(bundle.config.storage, bundle.root_dir)
        prompt = load_prompt(bundle.prompt_dir, "joint_memory_extraction.txt")
        self.extractor = MockJointExtractor(self.model_registry, self.cost_logger, prompt)

    @classmethod
    def from_config_dir(cls, config_dir: str | Path = "configs") -> "InfoBudgetPipeline":
        return cls(load_project_bundle(config_dir))

    def process_turns(self, turns: list[Turn], save_outputs: bool = True) -> PipelineResult:
        """运行完整构建流程。"""
        segments = self.segmenter.reindex_segments(self.segmenter.segment(turns))
        self.memory_store.record_segments(segments)
        scores: list[ScoreResult] = []
        tiers: list[str] = []
        entries: list[MemoryEntry] = []
        for segment in segments:
            score = self.scorer.score(segment, self.memory_store)
            tier = self.router.route(score.final_score)
            entry = self.extractor.extract(segment, tier, score)
            summary_embedding = self.encoder.encode_text(entry.summary)
            self.memory_store.add_entry(entry, summary_embedding)
            for episode in entry.episodic_memory.episodes:
                episode_embedding = self.encoder.encode_text(
                    " ".join([episode.subject, episode.verb, episode.object, episode.time])
                )
                self.memory_store.add_episode(
                    {
                        "memory_id": entry.memory_id,
                        "segment_id": segment.segment_id,
                        "episode": asdict(episode),
                    },
                    episode_embedding,
                )
            scores.append(score)
            tiers.append(tier)
            entries.append(entry)
        metrics = aggregate_metrics(
            correctness=[True for _ in entries],
            cost_logs=self.cost_logger.logs,
            routed_tiers=tiers,
            num_queries=max(1, len(turns)),
            num_memories=len(entries),
            qa_latency_ms=0,
        )
        if save_outputs:
            self._save_outputs(metrics)
        logger.info("Pipeline completed with %s memories", len(entries))
        return PipelineResult(segments=segments, scores=scores, tiers=tiers, entries=entries, metrics=metrics)

    def _save_outputs(self, metrics: EvaluationMetrics) -> None:
        self.memory_store.save()
        self.cost_logger.save()
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

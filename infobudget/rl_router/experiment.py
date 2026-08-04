"""End-to-end training loop over frozen candidates and real S assemblies."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from infobudget.rl_router.assembly import AssemblyManager
from infobudget.rl_router.evaluation import AssemblyEvaluator
from infobudget.rl_router.ledger import SqliteLedger
from infobudget.rl_router.router import EmbeddingMLPRouter, FeatureScaler
from infobudget.rl_router.schemas import ReplayResult, Tier, TopicSegment
from infobudget.rl_router.training import ConstrainedActorCriticTrainer, TrainingStep

VirtualCostFunction = Callable[[list[Tier]], ReplayResult]
CostNormalizer = Callable[[float], float]


class RLExperimentTrainer:
    """Train without re-extracting: actions copy candidates into S, then QA reads S."""

    def __init__(
        self,
        *,
        model: EmbeddingMLPRouter,
        scaler: FeatureScaler,
        trainer: ConstrainedActorCriticTrainer,
        assembly_manager: AssemblyManager,
        evaluator: AssemblyEvaluator,
        virtual_cost: VirtualCostFunction | None,
        output_dir: str | Path,
    ):
        self.model, self.scaler, self.trainer = model, scaler, trainer
        self.assembly_manager, self.evaluator, self.virtual_cost = assembly_manager, evaluator, virtual_cost
        self.output_dir = Path(output_dir)
        self.episodes = SqliteLedger(
            self.output_dir / "training_ledger.sqlite3",
            "episodes",
            ("episode_id",),
            legacy_jsonl_path=self.output_dir / "episodes.jsonl",
        )
        self.best_score = float("-inf")
        self.global_step = 0
        self._best_assembly = None
        self._best_scope: tuple[str, str] | None = None

    def train_sample(
        self,
        *,
        features: np.ndarray,
        segments: list[TopicSegment],
        questions: list[dict],
        steps: int,
        policy_version: str,
        sample_metadata: dict | None = None,
        virtual_cost: VirtualCostFunction | None = None,
        cost_normalizer: CostNormalizer | None = None,
        candidate_extraction_run_id: str | None = None,
        save_final: bool = True,
        track_best: bool = True,
    ) -> list[TrainingStep]:
        if not segments:
            raise ValueError("training sample has no segments")
        first = segments[0]
        cost_function = virtual_cost or self.virtual_cost
        if cost_function is None:
            raise ValueError("train_sample requires a virtual cost function")
        normalize = cost_normalizer or (lambda value: value)
        history: list[TrainingStep] = []
        for _ in range(steps):
            episode_id = f"{first.sample_id}:episode_{self.global_step:08d}"
            self.global_step += 1
            state: dict = {}

            def evaluate(actions: list[Tier]) -> tuple[float, float]:
                assembly = self.assembly_manager.create(
                    dataset_name=first.dataset_name,
                    split=first.split,
                    sample_id=first.sample_id,
                    segments=segments,
                    actions=actions,
                    probabilities=None,
                    episode_id=episode_id,
                    policy_version=policy_version,
                    router_type="embedding_mlp",
                    candidate_extraction_run_id=candidate_extraction_run_id,
                )
                if assembly.status != "ready":
                    raise RuntimeError(f"assembly failed for {episode_id}")
                qa_score, evaluations = self.evaluator.evaluate_sample(
                    questions,
                    dataset_name=first.dataset_name,
                    split=first.split,
                    sample_id=first.sample_id,
                    assembly_id=assembly.assembly_id,
                    sample_metadata=sample_metadata,
                )
                cost = cost_function(actions)
                state.update(assembly=assembly, evaluations=evaluations, cost=cost)
                return qa_score, normalize(cost.total_cost)

            step = self.trainer.step(features, evaluate)
            history.append(step)
            assembly = state["assembly"]
            cost = state["cost"]
            self.episodes.append(
                {
                    "episode_id": episode_id,
                    "sample_id": first.sample_id,
                    "assembly_id": assembly.assembly_id,
                    "route_decisions": step.actions,
                    "tier_counts": {tier: step.actions.count(tier) for tier in ("small", "medium", "large")},
                    "s_fact_count": assembly.point_count,
                    "qa_ids": [question.get("question_id") for question in questions],
                    "qa_score": step.qa_score,
                    "virtual_extraction_cost": cost.total_cost,
                    "normalized_cost": step.virtual_cost,
                    "virtual_input_tokens": cost.input_tokens,
                    "virtual_output_tokens": cost.output_tokens,
                    "reward": step.reward,
                    "lambda": step.lagrange_multiplier,
                    "router_type": "embedding_mlp",
                    "candidate_extraction_run_id": candidate_extraction_run_id,
                }
            )
            if track_best and step.qa_score > self.best_score and step.virtual_cost <= self.trainer.budget:
                if self._best_assembly is not None and self._best_scope is not None:
                    self.assembly_manager.cleanup(
                        self._best_assembly,
                        dataset_name=self._best_scope[0],
                        split=self._best_scope[1],
                    )
                self.best_score = step.qa_score
                self._best_assembly = assembly
                self._best_scope = (first.dataset_name, first.split)
                self.model.save_checkpoint(
                    self.output_dir / "checkpoints" / "best.pt",
                    self.scaler,
                    {
                        "episode_id": episode_id,
                        "qa_score": step.qa_score,
                        "normalized_cost": step.virtual_cost,
                        "virtual_extraction_cost": cost.total_cost,
                    },
                )
            else:
                self.assembly_manager.cleanup(assembly, dataset_name=first.dataset_name, split=first.split)
        if save_final:
            self.save_final()
        return history

    def save_final(self, metadata: dict | None = None) -> Path:
        return self.model.save_checkpoint(
            self.output_dir / "checkpoints" / "final.pt",
            self.scaler,
            {
                "steps": self.global_step,
                "lagrange_multiplier": self.trainer.lagrange_multiplier,
                "best_score": self.best_score,
                **(metadata or {}),
            },
        )

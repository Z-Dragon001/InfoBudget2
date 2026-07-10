"""Dataset-level memory build and evaluation runners."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from infobudget.config import ProjectBundle
from infobudget.datasets.loader import DatasetLoader
from infobudget.evaluation.answer_generation import DatasetAnswerGenerator
from infobudget.evaluation.judges import JudgeRegistry
from infobudget.evaluation.metrics import EvaluationMetrics, aggregate_metrics
from infobudget.evaluation.storage import EvaluationArtifactStore
from infobudget.memory.store import MemoryStore
from infobudget.retrieval.retriever import Retriever
from infobudget.runtime.pipeline import InfoBudgetPipeline
from infobudget.schemas import CostLogEntry, DatasetDialogueExample, RetrievalTrace, Tier
from infobudget.scoring.modes import ScoringMode, normalize_scoring_mode
from infobudget.utils.embeddings import build_text_encoder
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DatasetEvaluationResult:
    """Dataset evaluation result."""

    dataset_name: str
    split: str
    metrics: EvaluationMetrics
    predictions: list[dict]
    retrieval_traces: list[RetrievalTrace]


@dataclass(slots=True)
class DatasetMemoryBuildResult:
    """Dataset memory build result."""

    dataset_name: str
    split: str
    scoring_mode: ScoringMode
    memory_root: Path
    num_examples: int
    num_memories: int
    cost_logs: list[CostLogEntry]
    tiers: list[Tier]


class DatasetEvaluationRunner:
    """Build and evaluate dataset memories.

    The preferred experiment design is two-stage:
    1. build_memories(): extract and persist memories only.
    2. evaluate_existing_memories(): load persisted memories and run QA only.

    evaluate() remains as a compatibility wrapper that runs both stages.
    """

    def __init__(self, bundle: ProjectBundle, scoring_mode: ScoringMode = "full"):
        self.bundle = bundle
        self.scoring_mode = normalize_scoring_mode(scoring_mode)
        self.loader = DatasetLoader(bundle.config.dataset, bundle.root_dir)
        self.artifact_store = EvaluationArtifactStore(bundle.root_dir)
        self.encoder = build_text_encoder(bundle.config.segmentation.embedding_model)
        self.answer_generator = DatasetAnswerGenerator(bundle)

    def evaluate(
        self,
        dataset_name: str,
        split: str,
        sample_ids: set[str] | None = None,
        limit: int | None = None,
    ) -> DatasetEvaluationResult:
        """Compatibility wrapper: build memories, then evaluate the persisted memories."""
        self.build_memories(dataset_name, split, sample_ids=sample_ids, limit=limit)
        return self.evaluate_existing_memories(dataset_name, split, sample_ids=sample_ids, limit=limit)

    def build_memories(
        self,
        dataset_name: str,
        split: str,
        sample_ids: set[str] | None = None,
        limit: int | None = None,
    ) -> DatasetMemoryBuildResult:
        """Build and persist memories without running QA evaluation."""
        manifest = self.loader.load_manifest(dataset_name, split)
        memory_root = self.memory_root(dataset_name, split)
        selected_examples = 0
        processed_sample_ids: list[str] = []
        all_cost_logs: list[CostLogEntry] = []
        all_tiers: list[Tier] = []
        total_memories = 0

        for example in self.loader.iter_samples(dataset_name, split):
            if sample_ids and example.sample_id not in sample_ids:
                continue
            if limit is not None and selected_examples >= limit:
                break
            selected_examples += 1
            processed_sample_ids.append(example.sample_id)

            pipeline = InfoBudgetPipeline(
                self.bundle,
                self.scoring_mode,
                run_output_dir=self.sample_memory_dir(dataset_name, split, example.sample_id),
            )
            build_result = pipeline.process_turns(example.dialogue, save_outputs=False)
            pipeline.save_memory_outputs()

            total_memories += len(build_result.entries)
            all_cost_logs.extend(pipeline.cost_logger.logs)
            all_tiers.extend(build_result.tiers)

        num_examples = selected_examples if (sample_ids or limit is not None) else int(manifest.get("num_samples", 0))
        self._save_memory_build_manifest(
            dataset_name=dataset_name,
            split=split,
            memory_root=memory_root,
            num_examples=num_examples,
            processed_sample_ids=processed_sample_ids,
            sample_ids=sample_ids,
            limit=limit,
            num_memories=total_memories,
            cost_logs=all_cost_logs,
            tiers=all_tiers,
        )
        return DatasetMemoryBuildResult(
            dataset_name=dataset_name,
            split=split,
            scoring_mode=self.scoring_mode,
            memory_root=memory_root,
            num_examples=num_examples,
            num_memories=total_memories,
            cost_logs=all_cost_logs,
            tiers=all_tiers,
        )

    def evaluate_existing_memories(
        self,
        dataset_name: str,
        split: str,
        sample_ids: set[str] | None = None,
        limit: int | None = None,
    ) -> DatasetEvaluationResult:
        """Evaluate QA using previously persisted memories. No memory extraction is run here."""
        manifest = self.loader.load_manifest(dataset_name, split)
        predictions: list[dict] = []
        retrieval_traces: list[RetrievalTrace] = []
        all_cost_logs: list[CostLogEntry] = []
        correctness: list[bool] = []
        abstention_correctness: list[bool] = []
        evidence_hits: list[bool] = []
        evidence_recalls: list[float] = []
        retrieved_counts: list[int] = []
        group_labels: dict[str, list[bool]] = {}
        total_memories = 0
        selected_examples = 0
        memory_root = self.memory_root(dataset_name, split)

        for example in self.loader.iter_samples(dataset_name, split):
            if sample_ids and example.sample_id not in sample_ids:
                continue
            if limit is not None and selected_examples >= limit:
                break
            selected_examples += 1

            memory_store = self._load_memory_store(dataset_name, split, example.sample_id)
            retriever = Retriever(memory_store, self.encoder)
            total_memories += len(memory_store.entries)
            all_cost_logs.extend(self._load_cost_logs(dataset_name, split, example.sample_id))

            for qa_pair in example.qa_pairs:
                retrieval_top_k = self._retrieval_top_k(dataset_name)
                retrieved = retriever.retrieve(
                    qa_pair.question,
                    top_k=retrieval_top_k,
                )
                answer_result = self.answer_generator.generate(
                    dataset_name=dataset_name,
                    example=example,
                    qa_pair=qa_pair,
                    retrieved_entries=retrieved,
                )
                predicted_answer = answer_result.answer
                judge = JudgeRegistry.create(
                    qa_pair.judge_profile,
                    judge_mode=self.bundle.config.evaluation.judge_mode,
                    judge_model=self.bundle.config.evaluation.judge_model,
                )
                judge_result = judge.judge(qa_pair, predicted_answer, retrieved)
                trace = self._build_trace(example, qa_pair, retrieved, memory_store.segments)

                if self._counts_toward_main_accuracy(dataset_name, qa_pair):
                    correctness.append(judge_result.correct)
                if qa_pair.is_unanswerable:
                    abstention_correctness.append(judge_result.correct)
                evidence_hits.append(trace.evidence_hit)
                evidence_recalls.append(trace.evidence_recall_at_k)
                retrieved_counts.append(len(retrieved))

                group_name = qa_pair.category or qa_pair.question_type or qa_pair.judge_profile
                group_labels.setdefault(group_name, []).append(judge_result.correct)

                predictions.append(
                    {
                        "dataset_name": dataset_name,
                        "split": split,
                        "sample_id": example.sample_id,
                        "question_id": qa_pair.question_id,
                        "question": qa_pair.question,
                        "gold_answer": qa_pair.answer,
                        "predicted_answer": predicted_answer,
                        "correct": judge_result.correct,
                        "matched_by": judge_result.matched_by,
                        "answer_matched_by": answer_result.matched_by,
                        "qa_mode": self.bundle.config.evaluation.qa_mode,
                        "answer_model_tier": answer_result.answer_model_tier,
                        "answer_model_name": answer_result.model_name,
                        "answer_prompt": answer_result.prompt_name,
                        "answer_latency_ms": answer_result.latency_ms,
                        "judge_profile": qa_pair.judge_profile,
                        "question_type": qa_pair.question_type,
                        "category": qa_pair.category,
                        "is_unanswerable": qa_pair.is_unanswerable,
                        "retrieved_memory_ids": trace.retrieved_memory_ids,
                        "retrieved_segment_ids": trace.retrieved_segment_ids,
                        "evidence_hit": trace.evidence_hit,
                        "evidence_recall_at_k": trace.evidence_recall_at_k,
                    }
                )
                retrieval_traces.append(trace)

        num_examples = selected_examples if (sample_ids or limit is not None) else int(manifest.get("num_samples", 0))
        metrics = aggregate_metrics(
            correctness=correctness,
            cost_logs=all_cost_logs,
            routed_tiers=[item.tier for item in all_cost_logs],
            dataset_name=dataset_name,
            split=split,
            num_examples=num_examples,
            num_queries=len(correctness),
            num_memories=total_memories,
            evidence_hits=evidence_hits,
            evidence_recalls=evidence_recalls,
            retrieved_counts=retrieved_counts,
            abstention_correctness=abstention_correctness,
            group_labels=group_labels,
            qa_latency_ms=sum(item.get("answer_latency_ms", 0) for item in predictions),
        )
        self.artifact_store.save(
            dataset_name=dataset_name,
            split=self.evaluation_output_label(split),
            metrics=metrics,
            predictions=predictions,
            retrieval_traces=retrieval_traces,
            metadata={
                "dataset_name": dataset_name,
                "split": split,
                "output_label": self.evaluation_output_label(split),
                "scoring_mode": self.scoring_mode,
                "extraction_mode": self.bundle.config.extractor.extraction_mode,
                "num_examples": num_examples,
                "requested_sample_ids": sorted(sample_ids) if sample_ids else [],
                "sample_limit": limit,
                "num_predictions": len(predictions),
                "retrieval_top_k": self._retrieval_top_k(dataset_name),
                "save_retrieval_traces": self.bundle.config.evaluation.save_retrieval_traces,
                "qa_mode": self.bundle.config.evaluation.qa_mode,
                "answer_model_tier": self.bundle.config.evaluation.answer_model_tier,
                "answer_model_name": self._answer_model_name(),
                "qa_max_new_tokens": self.bundle.config.evaluation.qa_max_new_tokens,
                "judge_mode": self.bundle.config.evaluation.judge_mode,
                "judge_model": (
                    self.bundle.config.evaluation.judge_model.model_name
                    if self.bundle.config.evaluation.judge_model
                    else ""
                ),
                "judge_cost_counted": False,
                "memory_output_dir": str(memory_root),
                "memory_eval_mode": "evaluate_existing_memories",
            },
        )
        return DatasetEvaluationResult(
            dataset_name=dataset_name,
            split=split,
            metrics=metrics,
            predictions=predictions,
            retrieval_traces=retrieval_traces,
        )

    def memory_root(self, dataset_name: str, split: str) -> Path:
        """Return outputs/memory/{dataset}/{split}/{scoring_mode}/{extraction_mode}."""
        return (
            self.bundle.root_dir
            / "outputs"
            / "memory"
            / dataset_name
            / split
            / self.scoring_mode
            / self.bundle.config.extractor.extraction_mode
        )

    def sample_memory_dir(self, dataset_name: str, split: str, sample_id: str) -> Path:
        """Return the memory directory for one sample."""
        return self.memory_root(dataset_name, split) / sample_id

    def evaluation_output_label(self, split: str) -> str:
        """Return outputs/evaluation/{dataset}/{split}/{scoring_mode}/{extraction_mode} suffix."""
        return f"{split}/{self.scoring_mode}/{self.bundle.config.extractor.extraction_mode}"

    @staticmethod
    def _predict_answer(qa_pair, retrieved_entries: list) -> str:
        if not retrieved_entries:
            return "I don't know." if qa_pair.is_unanswerable else ""
        return retrieved_entries[0].memory

    def _retrieval_top_k(self, dataset_name: str) -> int:
        normalized = dataset_name.strip().lower()
        if normalized == "locomo":
            return self.bundle.config.evaluation.locomo_retrieval_top_k
        if normalized == "longmemeval":
            return self.bundle.config.evaluation.longmemeval_retrieval_top_k
        return self.bundle.config.evaluation.retrieval_top_k

    @staticmethod
    def _counts_toward_main_accuracy(dataset_name: str, qa_pair) -> bool:
        if dataset_name.strip().lower() == "locomo" and qa_pair.category == "category_5":
            return False
        return True

    def _answer_model_name(self) -> str:
        tier = self.bundle.config.evaluation.answer_model_tier
        model = self.bundle.models.get(tier)
        return model.effective_model_name if model else ""

    def _load_memory_store(self, dataset_name: str, split: str, sample_id: str) -> MemoryStore:
        storage_cfg = replace(self.bundle.config.storage, jsonl_dir="memory_jsonl", qdrant_dir="qdrant")
        store = MemoryStore(storage_cfg, self.sample_memory_dir(dataset_name, split, sample_id))
        store.load()
        if store.memory_index.is_empty():
            raise FileNotFoundError(
                f"missing Qdrant memory index for {dataset_name}/{split}/{self.scoring_mode}/{sample_id}; "
                "run scripts/build_dataset_memory.py first"
            )
        if store.needs_index_rebuild():
            logger.warning(
                "Qdrant memory index count (%s) differs from JSONL memory count (%s) for %s/%s/%s/%s; "
                "using Qdrant payloads for QA retrieval without rebuilding from JSONL",
                store.memory_index.count(),
                len(store.entries),
                dataset_name,
                split,
                self.scoring_mode,
                sample_id,
            )
        return store

    def _load_cost_logs(self, dataset_name: str, split: str, sample_id: str) -> list[CostLogEntry]:
        path = self.sample_memory_dir(dataset_name, split, sample_id) / "memory_jsonl" / "cost_logs.jsonl"
        if not path.exists():
            return []
        logs: list[CostLogEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    logs.append(CostLogEntry(**json.loads(line)))
        return logs

    def _save_memory_build_manifest(
        self,
        *,
        dataset_name: str,
        split: str,
        memory_root: Path,
        num_examples: int,
        processed_sample_ids: list[str],
        sample_ids: set[str] | None,
        limit: int | None,
        num_memories: int,
        cost_logs: list[CostLogEntry],
        tiers: list[Tier],
    ) -> None:
        memory_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset_name": dataset_name,
            "split": split,
            "scoring_mode": self.scoring_mode,
            "extraction_mode": self.bundle.config.extractor.extraction_mode,
            "memory_root": str(memory_root),
            "num_examples": num_examples,
            "processed_sample_ids": processed_sample_ids,
            "requested_sample_ids": sorted(sample_ids) if sample_ids else [],
            "sample_limit": limit,
            "num_memories": num_memories,
            "num_extraction_calls": len(cost_logs),
            "total_cost_usd": round(sum(item.cost_usd for item in cost_logs), 8),
            "routed_tier_counts": {tier: tiers.count(tier) for tier in ("small", "medium", "large")},
            "memory_build_stage": "build_memories_only",
        }
        with (memory_root / "build_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_trace(example: DatasetDialogueExample, qa_pair, retrieved_entries: list, segments: list) -> RetrievalTrace:
        retrieved_segment_ids = [entry.segment_id for entry in retrieved_entries]
        retrieved_summaries = [entry.memory for entry in retrieved_entries]
        retrieved_memory_ids = [entry.memory_id for entry in retrieved_entries]
        segment_lookup = {segment.segment_id: segment for segment in segments}
        session_turns = {
            session.session_id: {turn.turn_id for turn in session.turns}
            for session in example.sessions
        }

        matched_sessions: set[str] = set()
        for session in example.sessions:
            turn_ids = {turn.turn_id for turn in session.turns}
            if turn_ids & set(qa_pair.evidence_turn_ids):
                matched_sessions.add(session.session_id)
            if session.session_id in qa_pair.evidence_session_ids:
                matched_sessions.add(session.session_id)
        evidence_scope = set(qa_pair.evidence_session_ids) | matched_sessions
        evidence_hit = False
        if qa_pair.evidence_turn_ids:
            evidence_turn_set = set(qa_pair.evidence_turn_ids)
            for entry in retrieved_entries:
                source_turn_ids = _entry_turn_ids(entry, segment_lookup)
                if evidence_turn_set & source_turn_ids:
                    evidence_hit = True
                    break
        elif evidence_scope:
            for entry in retrieved_entries:
                source_turn_ids = _entry_turn_ids(entry, segment_lookup)
                segment_session_ids = {
                    session_id
                    for session_id, turn_ids in session_turns.items()
                    if turn_ids & source_turn_ids
                }
                if segment_session_ids & evidence_scope:
                    evidence_hit = True
                    break
        recall = 1.0 if not (qa_pair.evidence_turn_ids or evidence_scope) else (1.0 if evidence_hit else 0.0)
        return RetrievalTrace(
            question_id=qa_pair.question_id,
            sample_id=example.sample_id,
            retrieved_memory_ids=retrieved_memory_ids,
            retrieved_segment_ids=retrieved_segment_ids,
            retrieved_summaries=retrieved_summaries,
            evidence_session_ids=sorted(evidence_scope),
            evidence_turn_refs=qa_pair.evidence_turn_refs,
            evidence_hit=evidence_hit,
            evidence_recall_at_k=recall,
            metadata={
                "question_type": qa_pair.question_type,
                "category": qa_pair.category,
            },
        )


def _entry_turn_ids(entry, segment_lookup: dict) -> set[int]:
    if getattr(entry, "source_turn_ids", None):
        return {int(item) for item in entry.source_turn_ids}
    if getattr(entry, "source_turn_id", 0):
        return {int(entry.source_turn_id)}
    segment = segment_lookup.get(entry.segment_id)
    return set(segment.turn_ids) if segment else set()

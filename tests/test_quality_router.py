"""Tests for the supervised capability-conditioned quality-router path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from infobudget.quality_router.budget import optimize_budget
from infobudget.quality_router.config import QualityRouterConfig
from infobudget.quality_router.counterfactual import (
    aggregate_segment_usage,
    counterfactual_consistency,
)
from infobudget.quality_router.labeling import score_fact_sets
from infobudget.quality_router.model import (
    CapabilityConditionedQualityScorer,
    QualityFeatureBuilder,
)
from infobudget.quality_router.schemas import (
    CAPABILITY_DIMENSIONS,
    AtomicFact,
    FactSetKey,
    ModelCapabilityProfile,
    QualityPrediction,
)
from infobudget.rl_router.schemas import FactRecord, TopicSegment
from infobudget.rl_router.assembly import AssemblyManager
from infobudget.rl_router.ledger import read_sqlite_ledger
from infobudget.rl_router.qdrant_store import AssemblyResult
from infobudget.rl_router.router import FeatureScaler


class FakeEncoder:
    model_name = "fake-384"
    dimension = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(len(text))] + [0.0] * 383 for text in texts], dtype=np.float32
        )


def _profile(model_id: str = "model-a") -> ModelCapabilityProfile:
    return ModelCapabilityProfile.from_dict(
        {
            "profile_id": f"profile-{model_id}",
            "model_id": model_id,
            "dimensions": {name: 0.5 for name in CAPABILITY_DIMENSIONS},
            "benchmark_hash": "benchmark-hash",
            "evaluation_code_commit": "commit",
        }
    )


def _segment(segment_id: str = "segment-1") -> TopicSegment:
    return TopicSegment(
        dataset_name="dataset",
        split="train",
        sample_id="sample-1",
        session_id="session-1",
        segment_id=segment_id,
        segmentation_method="topic",
        segmentation_version="v1",
        start_turn=1,
        end_turn=2,
        turn_ids=(1, 2),
        start_timestamp=None,
        end_timestamp=None,
        text="Alice moved to Paris.",
        token_count=5,
        source_content_hash="hash",
    )


def test_quality_config_and_minilm_feature_dimension() -> None:
    config = QualityRouterConfig.load("configs/quality_router.yaml")
    assert config.values["label_name"] == "silver_strict_fact_f1"
    builder = QualityFeatureBuilder(FakeEncoder())
    builder.fit([_segment()])
    segment_features = builder.build_segment_features([_segment()])
    pair_features = builder.combine(segment_features, [_profile()])
    assert segment_features.shape == (1, 390)
    assert pair_features.shape == (1, 397)


def test_strict_fact_f1_uses_one_to_one_source_grounded_matching() -> None:
    references = [
        AtomicFact("r1", "Alice moved to Paris.", (1,)),
        AtomicFact("r2", "Bob lives in Rome.", (2,)),
    ]
    candidates = [
        AtomicFact("c1", "Alice moved to Paris.", (1,)),
        AtomicFact("c2", "Alice moved to Paris.", (1,)),
        AtomicFact("c3", "Bob lives in Rome.", (99,)),
    ]
    result = score_fact_sets(candidates, references, valid_source_turn_ids={1, 2})
    assert (result.true_positive, result.false_positive, result.false_negative) == (1, 2, 1)
    assert result.f1 == pytest.approx(0.4)


def test_empty_fact_sets_are_perfect_only_when_both_are_empty() -> None:
    assert score_fact_sets([], [], valid_source_turn_ids={1}).f1 == 1.0
    assert score_fact_sets([], [AtomicFact("r", "fact", (1,))], valid_source_turn_ids={1}).f1 == 0.0


def test_quality_checkpoint_freezes_embedding_and_memoryprint_schema(tmp_path) -> None:
    model = CapabilityConditionedQualityScorer(397, [16, 8], 0.0)
    checkpoint = model.save_checkpoint(
        tmp_path / "quality.pt",
        scaler=FeatureScaler([0.0] * 6, [1.0] * 6),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        metadata={"label_name": "silver_strict_fact_f1"},
    )
    restored, scaler, metadata = CapabilityConditionedQualityScorer.load_checkpoint(checkpoint)
    assert restored.input_dimension == 397
    assert scaler.scale == [1.0] * 6
    assert metadata["embedding_dimension"] == 384
    with torch.inference_mode():
        assert restored(torch.zeros((2, 397))).shape == (2,)


def test_fact_payload_round_trip_preserves_extra_provenance() -> None:
    fact = FactRecord(
        fact_id="fact-1",
        dataset_name="dataset",
        split="train",
        sample_id="sample-1",
        session_id="session-1",
        segment_id="segment-1",
        segment_hash="segment-hash",
        source_turn_ids=[1],
        fact_text="Alice moved to Paris.",
        fact_index=0,
        fact_count_in_segment=1,
        memory_tier="small",
        extractor_model="model-small",
        prompt_version="prompt-v1",
        batch_id="batch-1",
        extraction_run_id="run-1",
        segment_start_timestamp=None,
        segment_end_timestamp=None,
        allocated_input_tokens=10,
        allocated_output_tokens=5,
        allocated_total_tokens=15,
        allocated_input_cost=0.01,
        allocated_output_cost=0.02,
        allocated_total_cost=0.03,
        embedding_model="old-model",
        embedding_dimension=1024,
        extra={"model_family": "qwen", "audit_field": "kept"},
    )
    restored = FactRecord.from_payload(fact.payload())
    assert restored.segment_hash == "segment-hash"
    assert restored.extra["audit_field"] == "kept"
    assert restored.payload()["model_family"] == "qwen"
    assert restored.payload()["model_id"] == "model-small"


def test_budget_optimizer_selects_one_model_per_segment_and_respects_budget() -> None:
    first = FactSetKey("d", "test", "sample", "s1")
    second = FactSetKey("d", "test", "sample", "s2")
    predictions = [
        QualityPrediction(first, "small", "p-small", 0.50, 1.0),
        QualityPrediction(first, "large", "p-large", 0.90, 3.0),
        QualityPrediction(second, "small", "p-small", 0.60, 1.0),
        QualityPrediction(second, "large", "p-large", 0.95, 3.0),
    ]
    solution = optimize_budget(predictions, budget=4.0, cost_quantum=1.0)
    assert solution.total_cost == 4.0
    assert len(solution.selections) == 2
    assert {item.key.segment_id for item in solution.selections} == {"s1", "s2"}
    assert sum(item.model_id == "large" for item in solution.selections) == 1


def test_quality_route_metadata_is_persisted_in_assembly_ledger(tmp_path) -> None:
    class Store:
        def assemble(self, **kwargs):
            return AssemblyResult(
                "assembly-1",
                kwargs["sample_id"],
                "ready",
                1,
                {kwargs["segments"][0].segment_id: kwargs["actions"][0]},
            )

    manager = AssemblyManager(Store(), tmp_path / "routing.sqlite3")
    manager.create(
        dataset_name="dataset",
        split="train",
        sample_id="sample-1",
        segments=[_segment()],
        actions=["medium"],
        probabilities=None,
        episode_id="budget-run",
        policy_version="checkpoint",
        router_type="capability_conditioned_quality_v1",
        candidate_extraction_run_id="extract-run",
        route_metadata=[
            {
                "selected_model_id": "model-medium",
                "selected_profile_id": "profile-medium",
                "predicted_quality": 0.8,
                "selected_cost": 0.03,
                "route_decision_id": "decision-1",
                "quality_checkpoint_hash": "checkpoint-hash",
                "budget_run_id": "budget-run",
                "sample_budget": 0.5,
                "sample_total_selected_cost": 0.03,
            }
        ],
    )
    row = read_sqlite_ledger(tmp_path / "routing.sqlite3", "assemblies")[0]
    assert row["selected_model_id"] == "model-medium"
    assert row["predicted_quality"] == 0.8
    assert row["quality_checkpoint_hash"] == "checkpoint-hash"


def test_counterfactual_artifacts_keep_local_and_qa_scales_separate() -> None:
    usage = aggregate_segment_usage(
        [
            {
                "dataset": "d",
                "split": "test",
                "sample_id": "sample",
                "question_id": "q1",
                "category": "temporal",
                "correct": True,
                "retrieved": [{"segment_id": "s1"}, {"segment_id": "s1"}],
            },
            {
                "dataset": "d",
                "split": "test",
                "sample_id": "sample",
                "question_id": "q2",
                "category": "temporal",
                "correct": False,
                "retrieved": [{"segment_id": "s1"}],
            },
        ]
    )
    assert usage[0]["question_count"] == 2
    assert usage[0]["accuracy"] == 0.5
    metrics = counterfactual_consistency(
        [
            {"predicted_quality_delta": 0.1, "qa_delta": 0.4},
            {"predicted_quality_delta": -0.2, "qa_delta": -0.1},
        ]
    )
    assert metrics["sign_agreement"] == 1.0
    assert metrics["spearman"] == pytest.approx(1.0)

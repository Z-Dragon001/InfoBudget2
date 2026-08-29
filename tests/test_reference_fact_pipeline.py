from __future__ import annotations

import json
from pathlib import Path

import pytest

from infobudget.rl_router.api import LLMResponse
from infobudget.rl_router.schemas import TopicSegment
from infobudget.schemas import ModelSpec, PriceSpec
from reference_fact_pipeline.config import ReferencePipelineConfig
from reference_fact_pipeline.metrics import metrics_from_counts, score_fact_sets
from reference_fact_pipeline.pipeline import ReferenceFactPipeline


class QueueClient:
    def __init__(self, contents: list[dict[str, object]]):
        self.contents = list(contents)
        self.prompts: list[str] = []

    def complete(self, *, model_spec, prompt, max_new_tokens, json_mode):
        self.prompts.append(prompt)
        return LLMResponse(
            content=json.dumps(self.contents.pop(0)),
            input_tokens=100,
            output_tokens=20,
            latency_ms=5,
        )


def _config(max_reference_facts: int = 15) -> ReferencePipelineConfig:
    return ReferencePipelineConfig(
        schema_version="frozen_reference_fact_v1",
        prompt_version="test_v1",
        reference_extractor_role="reference",
        coverage_extractor_role="reference",
        grounding_judge_role="judge",
        candidate_roles=("small", "medium", "large"),
        require_non_candidate_reference_model=True,
        max_reference_facts=max_reference_facts,
        max_raw_facts=10,
        extraction_max_new_tokens=1000,
        coverage_max_new_tokens=1000,
        grounding_max_new_tokens=1000,
        timeout_seconds=30,
        max_retries=0,
        retry_backoff_seconds=0.0,
        fact_type_priority=("state", "event", "other"),
    )


def _model(name: str) -> ModelSpec:
    return ModelSpec(
        deploy="api",
        backend="openai_compatible",
        model_name=name,
        tokenizer_name=name,
        max_context_tokens=10000,
        tensor_parallel_size=1,
        dtype="n/a",
    )


def _segment() -> TopicSegment:
    return TopicSegment(
        dataset_name="locomo",
        split="full",
        sample_id="sample-1",
        session_id="session-1",
        segment_id="sample-1:seg-1",
        segmentation_method="test",
        segmentation_version="test-v1",
        start_turn=1,
        end_turn=2,
        turn_ids=(1, 2),
        start_timestamp=None,
        end_timestamp=None,
        text=(
            "[2023-01-01T00:00:00, Sun] 0.A: I moved to Paris.\n"
            "[2023-01-01T00:00:01, Sun] 1.B: Congratulations!"
        ),
        token_count=10,
        source_content_hash="source-hash",
    )


def test_metrics_implement_all_formulas():
    result = metrics_from_counts(2, 1, 2, beta=2.0)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(1 / 2)
    assert result.f1 == pytest.approx(4 / 7)
    assert result.f_beta == pytest.approx(10 / 19)
    assert result.jaccard == pytest.approx(2 / 5)
    assert result.false_discovery_rate == pytest.approx(1 / 3)
    assert result.false_negative_rate == pytest.approx(1 / 2)
    assert result.exact_set_match == 0.0


def test_matching_is_one_to_one_and_invalid_sources_are_false_posititives():
    result = score_fact_sets(
        ["c1", "c2", "bad"],
        ["r1", "r2"],
        [("c1", "r1"), ("c1", "r2"), ("c2", "r1"), ("bad", "r2")],
        invalid_candidate_ids=["bad"],
    )
    assert (result.tp, result.fp, result.fn) == (2, 1, 0)
    assert result.source_validity_rate == pytest.approx(2 / 3)


def test_pipeline_builds_stable_frozen_facts_and_annotates_source_ids(tmp_path: Path):
    client = QueueClient(
        [
            {
                "facts": [
                    {
                        "fact_text": "A moved to Paris",
                        "source_turn_ids": [1],
                        "fact_type": "state",
                        "state_status": "current",
                    }
                ]
            },
            {
                "decisions": [
                    {
                        "temp_fact_id": "initial_0000",
                        "decision": "ACCEPT",
                        "entailed": True,
                        "atomic": True,
                        "source_ids_sufficient": True,
                        "contains_external_inference": False,
                        "duplicate_of": None,
                        "reason": "turn 1 says so",
                    }
                ]
            },
            {
                "missing_facts": [
                    {
                        "fact_text": "B congratulated A",
                        "source_turn_ids": [2],
                        "fact_type": "event",
                        "state_status": "historical",
                    }
                ]
            },
            {
                "decisions": [
                    {
                        "temp_fact_id": "coverage_0000",
                        "decision": "ACCEPT",
                        "entailed": True,
                        "atomic": True,
                        "source_ids_sufficient": True,
                        "contains_external_inference": False,
                        "duplicate_of": None,
                        "reason": "turn 2 says so",
                    }
                ]
            },
        ]
    )
    models = {
        "reference": _model("reference-model"),
        "judge": _model("judge-model"),
        "small": _model("candidate-small"),
        "medium": _model("candidate-medium"),
        "large": _model("candidate-large"),
    }
    prices = {
        name: PriceSpec(official_price_in_per_1m=1.0, official_price_out_per_1m=2.0)
        for name in ("reference-model", "judge-model")
    }
    pipeline = ReferenceFactPipeline(
        config=_config(),
        models=models,
        prices=prices,
        client=client,
        prompt_dir=Path("reference_fact_pipeline/prompts"),
        raw_archive_dir=tmp_path / "raw",
    )
    result = pipeline.process_segment(_segment(), run_id="test-run")
    assert result.frozen_fact_count == 2
    assert len(result.reference_facts[0].reference_fact_id) == 23
    assert result.reference_set_hash
    assert result.to_dict()["total_cost"] > 0
    assert "<SOURCE_TURN_ID=1> [2023-01-01T00:00:00, Sun] 0.A" in client.prompts[0]
    assert (tmp_path / "raw").is_dir()


def test_reference_model_cannot_equal_candidate_model():
    models = {
        "reference": _model("same-model"),
        "judge": _model("judge-model"),
        "small": _model("same-model"),
        "medium": _model("candidate-medium"),
        "large": _model("candidate-large"),
    }
    with pytest.raises(ValueError, match="candidate model"):
        ReferenceFactPipeline(
            config=_config(),
            models=models,
            prices={},
            client=QueueClient([]),
            prompt_dir=Path("reference_fact_pipeline/prompts"),
        )


def test_source_annotation_preserves_multiline_turns(tmp_path: Path):
    segment = _segment()
    segment = TopicSegment(
        **{
            name: getattr(segment, name)
            for name in segment.__dataclass_fields__
            if name != "text"
        },
        text=(
            "[2023-01-01T00:00:00, Sun] 0.user: First paragraph.\n\nSecond paragraph.\n"
            "[2023-01-01T00:00:01, Sun] 1.assistant: Answer."
        ),
    )
    client = QueueClient(
        [
            {"facts": []},
            {"missing_facts": []},
        ]
    )
    models = {
        "reference": _model("reference-model"),
        "judge": _model("judge-model"),
        "small": _model("candidate-small"),
        "medium": _model("candidate-medium"),
        "large": _model("candidate-large"),
    }
    prices = {
        name: PriceSpec(official_price_in_per_1m=1.0, official_price_out_per_1m=2.0)
        for name in ("reference-model", "judge-model")
    }
    pipeline = ReferenceFactPipeline(
        config=_config(),
        models=models,
        prices=prices,
        client=client,
        prompt_dir=Path("reference_fact_pipeline/prompts"),
        raw_archive_dir=tmp_path,
    )
    pipeline.process_segment(segment, run_id="multiline")
    assert "Second paragraph" in client.prompts[0]
    assert "<SOURCE_TURN_ID=2> [2023-01-01T00:00:01" in client.prompts[0]

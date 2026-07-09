"""Tests for OpenAI-compatible joint memory extraction."""

from __future__ import annotations

import json
from pathlib import Path

from infobudget.config import load_project_bundle
from infobudget.cost.logger import CostLogger
from infobudget.extractors.llm_joint import (
    APIJointExtractor,
    LLMResponse,
    LocalJointExtractor,
    TieredJointExtractor,
)
from infobudget.extractors.mock_joint import MockJointExtractor
from infobudget.runtime.model_registry import ModelRegistry, PriceRegistry
from infobudget.runtime.prompt_loader import load_prompt_map
from infobudget.schemas import ModelSpec, ScoreResult, Segment


class FakeClient:
    """Fake chat-completion client that returns a valid memory payload."""

    def __init__(self) -> None:
        self.models: list[str] = []
        self.prompts: list[str] = []

    def complete(
        self,
        *,
        model_spec: ModelSpec,
        prompt: str,
        max_new_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        self.models.append(model_spec.model_name)
        self.prompts.append(prompt)
        if "relation" in prompt and "Relational Memory Extractor" in prompt:
            payload = {
                "data": [
                    {
                        "source_id": 0,
                        "relation": "User proposed routing memory extraction by information score.",
                    },
                ],
            }
        else:
            payload = {
                "data": [
                    {
                        "source_id": 0,
                        "fact": "User said InfoBudget should route segments to model tiers.",
                    },
                ],
            }
        return LLMResponse(
            content=json.dumps(payload),
            input_tokens=111,
            output_tokens=77,
            latency_ms=42,
        )


def test_tiered_joint_extractor_maps_llm_json_and_uses_api_tier(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    registry = ModelRegistry(bundle.models)
    cost_logger = CostLogger(PriceRegistry(bundle.prices), tmp_path / "cost_logs.jsonl")
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
    local_client = FakeClient()
    api_client = FakeClient()
    extractor = TieredJointExtractor(
        model_registry=registry,
        local_extractor=LocalJointExtractor(
            registry,
            cost_logger,
            prompts,
            relational_prompt_template=relational_prompts,
            client=local_client,
        ),
        api_extractor=APIJointExtractor(
            registry,
            cost_logger,
            prompts,
            relational_prompt_template=relational_prompts,
            client=api_client,
        ),
        fallback_extractor=MockJointExtractor(registry, cost_logger, prompts),
        fallback_on_error=False,
    )
    segment = Segment(
        segment_id="seg_1",
        start_turn=1,
        end_turn=1,
        turn_ids=[1],
        text="[2026-07-01, Wed] 0.User: InfoBudget should route segments to model tiers.",
        token_count=12,
        mean_adjacent_similarity=1.0,
        boundary_reason="test",
    )

    entries = extractor.extract(
        segment,
        "large",
        ScoreResult(intrinsic_score=0.8, utility_score=0.9, final_score=0.85, details={}),
    )

    assert api_client.models == [bundle.models["large"].model_name]
    assert local_client.models == []
    assert len(entries) == 1
    assert entries[0].memory == "User said InfoBudget should route segments to model tiers."
    assert entries[0].entry_type == "factual"
    assert entries[0].time_stamp == "2026-07-01"
    assert entries[0].weekday == "Wed"
    assert entries[0].speaker_name == "User"
    assert entries[0].topic_id == 0
    assert cost_logger.logs[0].backend == "api"
    assert cost_logger.logs[0].extraction_mode == "flat_factual"


def test_event_mode_calls_factual_and_relational_prompts(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    registry = ModelRegistry(bundle.models)
    cost_logger = CostLogger(PriceRegistry(bundle.prices), tmp_path / "cost_logs.jsonl")
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
    client = FakeClient()
    extractor = APIJointExtractor(
        registry,
        cost_logger,
        prompts,
        relational_prompt_template=relational_prompts,
        client=client,
        extraction_mode="event",
    )
    segment = Segment(
        segment_id="seg_1",
        start_turn=1,
        end_turn=1,
        turn_ids=[1],
        text="[2026-07-01, Wed] 0.User: InfoBudget should route segments to model tiers.",
        token_count=12,
        mean_adjacent_similarity=1.0,
        boundary_reason="test",
    )

    entries = extractor.extract(
        segment,
        "large",
        ScoreResult(intrinsic_score=0.8, utility_score=0.9, final_score=0.85, details={}),
    )

    assert client.models == [bundle.models["large"].model_name, bundle.models["large"].model_name]
    assert [entry.entry_type for entry in entries] == ["factual", "relational"]
    assert [log.extraction_mode for log in cost_logger.logs] == ["event_factual", "event_relational"]

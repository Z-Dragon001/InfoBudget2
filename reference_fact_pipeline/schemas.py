"""Versioned contracts for raw, judged, and frozen reference Facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FactType = Literal[
    "identity",
    "state",
    "event",
    "plan",
    "preference",
    "goal",
    "relationship",
    "decision",
    "constraint",
    "health",
    "knowledge",
    "assistant_answer",
    "negative",
    "other",
]
StateStatus = Literal["current", "historical", "timeless", "unspecified"]


@dataclass(frozen=True, slots=True)
class ProposedFact:
    temp_fact_id: str
    fact_text: str
    source_turn_ids: tuple[int, ...]
    fact_type: str = "other"
    state_status: str = "unspecified"
    origin: str = "initial"
    proposal_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_turn_ids"] = list(self.source_turn_ids)
        return value


@dataclass(frozen=True, slots=True)
class GroundingDecision:
    temp_fact_id: str
    decision: Literal["ACCEPT", "REJECT"]
    entailed: bool
    atomic: bool
    source_ids_sufficient: bool
    contains_external_inference: bool
    duplicate_of: str | None
    reason: str

    @property
    def accepted(self) -> bool:
        return (
            self.decision == "ACCEPT"
            and self.entailed
            and self.atomic
            and self.source_ids_sufficient
            and not self.contains_external_inference
            and not self.duplicate_of
        )


@dataclass(frozen=True, slots=True)
class FrozenReferenceFact:
    reference_fact_id: str
    fact_text: str
    source_turn_ids: tuple[int, ...]
    fact_type: str
    state_status: str
    origin: str
    grounding_reason: str
    selection_rank: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_turn_ids"] = list(self.source_turn_ids)
        value["text"] = value["fact_text"]
        return value


@dataclass(slots=True)
class StageUsage:
    stage: str
    role: str
    configured_model: str
    request_model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    usage_source: str
    retry_count: int
    latency_ms: int
    provider_request_id: str = ""
    finish_reason: str = ""
    cost_status: str = "known"

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["total_tokens"] = self.input_tokens + self.output_tokens
        value["total_cost"] = self.total_cost
        return value


@dataclass(slots=True)
class FrozenReferenceSet:
    schema_version: str
    dataset_name: str
    split: str
    sample_id: str
    session_id: str
    segment_id: str
    segment_order: int
    segmentation_method: str
    segmentation_version: str
    source_content_hash: str
    segment_turn_ids: tuple[int, ...]
    reference_set_hash: str
    reference_facts: list[FrozenReferenceFact]
    rejected_facts: list[dict[str, Any]]
    raw_proposal_count: int
    grounded_accept_count: int
    frozen_fact_count: int
    truncated_to_k: bool
    run_id: str
    prompt_version: str
    config_hash: str
    stage_usage: list[StageUsage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dataset"] = self.dataset_name
        value["segment_turn_ids"] = list(self.segment_turn_ids)
        value["reference_facts"] = [item.to_dict() for item in self.reference_facts]
        value["stage_usage"] = [item.to_dict() for item in self.stage_usage]
        value["total_input_tokens"] = sum(
            item.input_tokens for item in self.stage_usage
        )
        value["total_output_tokens"] = sum(
            item.output_tokens for item in self.stage_usage
        )
        value["total_tokens"] = (
            value["total_input_tokens"] + value["total_output_tokens"]
        )
        value["provider_usage_stage_count"] = sum(
            item.usage_source == "provider" for item in self.stage_usage
        )
        value["estimated_usage_stage_count"] = sum(
            item.usage_source != "provider" for item in self.stage_usage
        )
        value["total_cost"] = sum(item.total_cost for item in self.stage_usage)
        value["cost_complete"] = all(
            item.cost_status == "known" for item in self.stage_usage
        )
        value["unknown_cost_stage_count"] = sum(
            item.cost_status != "known" for item in self.stage_usage
        )
        return value

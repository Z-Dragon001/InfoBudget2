"""Data contracts for fact extraction, storage, and routing."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, Literal

Tier = Literal["small", "medium", "large"]
TIERS: tuple[Tier, ...] = ("small", "medium", "large")


@dataclass(frozen=True, slots=True)
class TopicSegment:
    dataset_name: str
    split: str
    sample_id: str
    session_id: str
    segment_id: str
    segmentation_method: str
    segmentation_version: str
    start_turn: int
    end_turn: int
    turn_ids: tuple[int, ...]
    start_timestamp: str | None
    end_timestamp: str | None
    text: str
    token_count: int
    source_content_hash: str
    segment_order: int = 0
    extraction_truncated: bool = False
    extraction_original_char_count: int = 0
    extraction_retained_char_count: int = 0
    extraction_visible_source_ids: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TopicSegment":
        required = {
            item.name
            for item in cls.__dataclass_fields__.values()
            if item.default is MISSING and item.default_factory is MISSING
        }
        missing = sorted(name for name in required if name not in value)
        if missing:
            raise ValueError(f"segment is missing required fields: {', '.join(missing)}")
        payload = {
            name: value[name]
            for name in cls.__dataclass_fields__
            if name in value
        }
        payload["turn_ids"] = tuple(int(item) for item in value["turn_ids"])
        payload["segment_order"] = int(value.get("segment_order", value.get("segment_index", 0)))
        payload["extraction_truncated"] = bool(value.get("extraction_truncated", False))
        payload["extraction_original_char_count"] = int(
            value.get("extraction_original_char_count", 0)
        )
        payload["extraction_retained_char_count"] = int(
            value.get("extraction_retained_char_count", 0)
        )
        payload["extraction_visible_source_ids"] = tuple(
            int(item) for item in value.get("extraction_visible_source_ids", ())
        )
        return cls(**payload)


@dataclass(slots=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int
    model_name: str
    usage_source: str = "provider"
    retry_count: int = 0
    latency_ms: int = 0
    provider_request_id: str = ""
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class SegmentAllocation:
    segment_id: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    fact_count: int
    serialized_input_tokens: int
    attributed_output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass(slots=True)
class FactRecord:
    fact_id: str
    dataset_name: str
    split: str
    sample_id: str
    session_id: str
    segment_id: str
    segment_hash: str
    source_turn_ids: list[int]
    fact_text: str
    fact_index: int
    fact_count_in_segment: int
    memory_tier: Tier
    extractor_model: str
    prompt_version: str
    batch_id: str
    extraction_run_id: str
    segment_start_timestamp: str | None
    segment_end_timestamp: str | None
    allocated_input_tokens: float
    allocated_output_tokens: float
    allocated_total_tokens: float
    allocated_input_cost: float
    allocated_output_cost: float
    allocated_total_cost: float
    embedding_model: str
    embedding_dimension: int
    segment_order: int = 0
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("extra")
        value.update(self.extra)
        value["schema_version"] = "qdrant_fact_v2"
        value["source_content_hash"] = value.pop("segment_hash")
        return value


@dataclass(slots=True)
class BatchCompletion:
    content: str
    usage: ProviderUsage
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ParsedBatch:
    facts_by_segment: dict[str, list[str]]
    source_ids_by_segment: dict[str, list[list[int]]]
    block_text_by_segment: dict[str, str]


@dataclass(slots=True)
class ReplaySegmentCost:
    segment_id: str
    tier: Tier
    serialized_input_tokens: int
    attributed_output_tokens: int


@dataclass(slots=True)
class ReplayResult:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    batch_count_by_tier: dict[str, int]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

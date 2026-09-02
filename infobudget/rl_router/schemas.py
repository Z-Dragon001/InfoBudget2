"""Data contracts for fact extraction, storage, and routing."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, Literal

Tier = Literal["small", "medium", "large"]
TIERS: tuple[Tier, ...] = ("small", "medium", "large")
QDRANT_FACT_SCHEMA_VERSION = "qdrant_fact_v3"
SEGMENT_AUDIT_SCHEMA_VERSION = "segment_extraction_audit_v1"
SEGMENT_AUDIT_REQUIRED_FIELDS: tuple[str, ...] = (
    "audit_schema_version",
    "dataset_name",
    "split",
    "sample_id",
    "session_id",
    "model_family",
    "campaign_id",
    "campaign_scope_hash",
    "extraction_scope_hash",
    "qdrant_namespace",
    "segmentation_method",
    "segmentation_version",
    "source_content_hash",
    "segment_order",
    "segment_start_turn",
    "segment_end_turn",
    "segment_turn_ids",
    "segment_start_timestamp",
    "segment_end_timestamp",
    "segment_turn_count",
    "segment_token_count",
    "segment_char_count",
    "extractor_configured_model",
    "extractor_request_model",
    "extractor_backend",
    "prompt_version",
    "prompt_sha256",
    "embedding_model",
    "embedding_dimension",
    "embedding_model_hash",
    "embedding_revision",
    "embedding_normalized",
    "qdrant_distance",
    "input_price_per_1m",
    "output_price_per_1m",
    "price_effective_date",
    "currency",
    "extraction_run_id",
    "batch_id",
    "segment_id",
    "tier",
    "allocated_input_tokens",
    "allocated_output_tokens",
    "allocated_total_tokens",
    "allocated_input_cost",
    "allocated_output_cost",
    "allocated_total_cost",
    "fact_count",
    "status",
)


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
    model_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> "FactRecord":
        """Reconstruct a FactRecord without discarding versioned provenance fields."""
        payload = dict(value)
        if "segment_hash" not in payload and "source_content_hash" in payload:
            payload["segment_hash"] = payload["source_content_hash"]
        field_names = set(cls.__dataclass_fields__)
        constructor = {
            name: payload[name]
            for name in field_names - {"extra"}
            if name in payload
        }
        required = {
            item.name
            for item in cls.__dataclass_fields__.values()
            if item.name != "extra"
            and item.default is MISSING
            and item.default_factory is MISSING
        }
        missing = sorted(required - constructor.keys())
        if missing:
            raise ValueError(f"fact payload is missing required fields: {', '.join(missing)}")
        constructor["source_turn_ids"] = [int(item) for item in constructor["source_turn_ids"]]
        consumed = field_names | {"schema_version", "source_content_hash"}
        constructor["extra"] = {
            name: item for name, item in payload.items() if name not in consumed
        }
        return cls(**constructor)

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("extra")
        value.update(self.extra)
        value["model_id"] = str(value.get("model_id") or self.extractor_model)
        value["schema_version"] = QDRANT_FACT_SCHEMA_VERSION
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
    normalizations: list[dict[str, Any]] = field(default_factory=list)


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

"""Versioned data contracts for capability-conditioned quality routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CAPABILITY_DIMENSIONS: tuple[str, ...] = (
    "memory_recall",
    "target_memory_precision",
    "memory_accuracy",
    "current_state_accuracy",
    "target_binding_accuracy",
    "stale_value_rejection",
    "evidence_f1",
)


def _non_empty(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


@dataclass(frozen=True, slots=True)
class ModelCapabilityProfile:
    profile_id: str
    model_id: str
    dimensions: dict[str, float]
    benchmark_hash: str
    evaluation_code_commit: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelCapabilityProfile":
        dimensions = value.get("dimensions")
        if not isinstance(dimensions, dict):
            raise ValueError("capability profile dimensions must be an object")
        missing = sorted(set(CAPABILITY_DIMENSIONS) - dimensions.keys())
        extra = sorted(set(dimensions) - set(CAPABILITY_DIMENSIONS))
        if missing or extra:
            raise ValueError(
                f"capability dimensions mismatch; missing={missing}, extra={extra}"
            )
        normalized = {name: float(dimensions[name]) for name in CAPABILITY_DIMENSIONS}
        invalid = {name: number for name, number in normalized.items() if not 0.0 <= number <= 1.0}
        if invalid:
            raise ValueError(f"capability values must be in [0, 1]: {invalid}")
        return cls(
            profile_id=_non_empty(value.get("profile_id"), "profile_id"),
            model_id=_non_empty(value.get("model_id"), "model_id"),
            dimensions=normalized,
            benchmark_hash=_non_empty(value.get("benchmark_hash"), "benchmark_hash"),
            evaluation_code_commit=_non_empty(
                value.get("evaluation_code_commit"), "evaluation_code_commit"
            ),
        )

    def vector(self) -> list[float]:
        return [self.dimensions[name] for name in CAPABILITY_DIMENSIONS]


@dataclass(frozen=True, slots=True)
class AtomicFact:
    fact_id: str
    text: str
    source_turn_ids: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, id_fields: tuple[str, ...]) -> "AtomicFact":
        fact_id = next((str(value.get(name) or "").strip() for name in id_fields if value.get(name)), "")
        return cls(
            fact_id=_non_empty(fact_id, "/".join(id_fields)),
            text=_non_empty(value.get("text") or value.get("fact_text") or value.get("fact"), "fact text"),
            source_turn_ids=tuple(sorted({int(item) for item in value.get("source_turn_ids", value.get("source_ids", ())) })),
        )


@dataclass(frozen=True, slots=True)
class FactSetKey:
    dataset: str
    split: str
    sample_id: str
    segment_id: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactSetKey":
        return cls(
            dataset=_non_empty(value.get("dataset") or value.get("dataset_name"), "dataset"),
            split=_non_empty(value.get("split"), "split"),
            sample_id=_non_empty(value.get("sample_id"), "sample_id"),
            segment_id=_non_empty(value.get("segment_id"), "segment_id"),
        )

    def tuple(self) -> tuple[str, str, str, str]:
        return self.dataset, self.split, self.sample_id, self.segment_id


@dataclass(frozen=True, slots=True)
class FactQualityLabel:
    key: FactSetKey
    model_id: str
    profile_id: str
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    silver_strict_fact_f1: float
    reference_set_hash: str
    candidate_extraction_run_id: str
    label_version: str = "silver_f1_v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        key = value.pop("key")
        value.update(key)
        value["tp"] = value.pop("true_positive")
        value["fp"] = value.pop("false_positive")
        value["fn"] = value.pop("false_negative")
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactQualityLabel":
        quality = float(value["silver_strict_fact_f1"])
        if not 0.0 <= quality <= 1.0:
            raise ValueError("silver_strict_fact_f1 must be in [0, 1]")
        return cls(
            key=FactSetKey.from_dict(value),
            model_id=_non_empty(value.get("model_id") or value.get("extractor_model"), "model_id"),
            profile_id=_non_empty(value.get("profile_id"), "profile_id"),
            true_positive=int(value.get("tp", value.get("true_positive", 0))),
            false_positive=int(value.get("fp", value.get("false_positive", 0))),
            false_negative=int(value.get("fn", value.get("false_negative", 0))),
            precision=float(value.get("precision", 0.0)),
            recall=float(value.get("recall", 0.0)),
            silver_strict_fact_f1=quality,
            reference_set_hash=_non_empty(value.get("reference_set_hash"), "reference_set_hash"),
            candidate_extraction_run_id=_non_empty(
                value.get("candidate_extraction_run_id"), "candidate_extraction_run_id"
            ),
            label_version=str(value.get("label_version") or "silver_f1_v1"),
        )


@dataclass(frozen=True, slots=True)
class QualityPrediction:
    key: FactSetKey
    model_id: str
    profile_id: str
    predicted_quality: float
    cost: float


@dataclass(frozen=True, slots=True)
class RouteDecision:
    key: FactSetKey
    selected_model_id: str
    selected_profile_id: str
    predicted_quality: float
    cost: float
    route_decision_id: str
    quality_checkpoint_hash: str
    budget_run_id: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        key = value.pop("key")
        value.update(key)
        return value

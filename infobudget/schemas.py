"""Shared data contracts for preprocessing, segmentation, and model configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Turn:
    turn_id: int
    role: str
    text: str
    token_count: int
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def memory_text(self) -> str:
        caption = self.metadata.get("blip_caption")
        if caption and not self.metadata.get("image_description_appended"):
            return f"{self.text} (image description: {caption})"
        return self.text

    def memory_line(self) -> str:
        weekday = self.metadata.get("weekday")
        if self.timestamp and weekday:
            return f"[{self.timestamp}, {weekday}] {self.turn_id - 1}.{self.role}: {self.memory_text()}"
        if self.timestamp:
            return f"[{self.timestamp}] {self.turn_id - 1}.{self.role}: {self.memory_text()}"
        return f"{self.role}: {self.memory_text()}"


@dataclass(slots=True)
class Segment:
    segment_id: str
    start_turn: int
    end_turn: int
    turn_ids: list[int]
    text: str
    token_count: int
    mean_adjacent_similarity: float
    boundary_reason: str


@dataclass(slots=True)
class DatasetQAPair:
    question_id: str
    question: str
    answer: str
    question_type: str = ""
    category: str = ""
    question_date: str | None = None
    evidence_turn_ids: list[int] = field(default_factory=list)
    evidence_turn_refs: list[str] = field(default_factory=list)
    evidence_session_ids: list[str] = field(default_factory=list)
    judge_profile: str = "generic"
    is_unanswerable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetSession:
    session_id: str
    timestamp: str | None
    raw_timestamp: str | None
    turns: list[Turn]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetDialogueExample:
    sample_id: str
    dataset_name: str
    split: str
    sessions: list[DatasetSession]
    dialogue: list[Turn]
    qa_pairs: list[DatasetQAPair]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelSpec:
    deploy: str
    backend: str
    model_name: str
    tokenizer_name: str
    max_context_tokens: int
    tensor_parallel_size: int
    dtype: str
    max_output_tokens: int = 0
    tokenizer_local_path: str = ""
    api_base_url: str = ""
    api_key_env: str = ""
    request_model_name: str = ""

    @property
    def effective_model_name(self) -> str:
        return self.request_model_name or self.model_name

    @property
    def max_input_tokens(self) -> int:
        """Largest prompt when reserving the configured maximum model output."""
        return self.max_context_tokens - self.max_output_tokens

    def resolved_api_key(self) -> str:
        return os.getenv(self.api_key_env, "") if self.api_key_env else ""


@dataclass(slots=True)
class PriceSpec:
    official_price_in_per_1m: float
    official_price_out_per_1m: float
    currency: str = "USD"
    price_effective_date: str = ""

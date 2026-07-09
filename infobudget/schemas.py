"""功能：定义 InfoBudget 全局数据结构。
输入：模块间传递的结构化数据。
输出：标准化 dataclass 与辅助方法。
依赖：标准库。
作者：OpenAI Codex
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Tier = Literal["small", "medium", "large"]


def utc_now_iso() -> str:
    """返回 UTC ISO8601 时间戳。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class Turn:
    """单轮对话。"""

    turn_id: int
    role: str
    text: str
    token_count: int
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def memory_text(self) -> str:
        """Return turn text enriched with lightweight multimodal cues for retrieval."""
        blip_caption = self.metadata.get("blip_caption")
        if blip_caption:
            return f"{self.text} (image description: {blip_caption})"
        return self.text

    def memory_line(self) -> str:
        """Render a LightMem-style memory line when timestamp metadata is available."""
        weekday = self.metadata.get("weekday")
        if self.timestamp and weekday:
            return f"[{self.timestamp}, {weekday}] {self.turn_id - 1}.{self.role}: {self.memory_text()}"
        if self.timestamp:
            return f"[{self.timestamp}] {self.turn_id - 1}.{self.role}: {self.memory_text()}"
        return f"{self.role}: {self.memory_text()}"


@dataclass(slots=True)
class Segment:
    """主题分段结果。"""

    segment_id: str
    start_turn: int
    end_turn: int
    turn_ids: list[int]
    text: str
    token_count: int
    mean_adjacent_similarity: float
    boundary_reason: str


@dataclass(slots=True)
class ScoreResult:
    """信息价值评分结果。"""

    intrinsic_score: float
    utility_score: float
    final_score: float
    details: dict[str, float]


@dataclass(slots=True)
class DatasetQAPair:
    """统一的问答样本。"""

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
        """转换为可序列化字典。"""
        return asdict(self)


@dataclass(slots=True)
class DatasetSession:
    """统一的对话 session。"""

    session_id: str
    timestamp: str | None
    raw_timestamp: str | None
    turns: list[Turn]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。"""
        return asdict(self)


@dataclass(slots=True)
class DatasetDialogueExample:
    """统一的数据集对话样本。"""

    sample_id: str
    dataset_name: str
    split: str
    sessions: list[DatasetSession]
    dialogue: list[Turn]
    qa_pairs: list[DatasetQAPair]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。"""
        return asdict(self)


@dataclass(slots=True)
class SemanticEntity:
    """语义实体。"""

    name: str
    type: str
    aliases: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SemanticFact:
    """语义事实。"""

    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass(slots=True)
class Preference:
    """偏好记忆。"""

    owner: str
    item: str
    value: str
    confidence: float


@dataclass(slots=True)
class Constraint:
    """约束记忆。"""

    owner: str
    rule: str
    confidence: float


@dataclass(slots=True)
class Episode:
    """情景记忆。"""

    subject: str
    verb: str
    object: str
    time: str
    location: str
    sequence: int
    cause: str
    result: str
    confidence: float


@dataclass(slots=True)
class SemanticMemory:
    """语义记忆集合。"""

    entities: list[SemanticEntity] = field(default_factory=list)
    facts: list[SemanticFact] = field(default_factory=list)
    preferences: list[Preference] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)


@dataclass(slots=True)
class EpisodicMemory:
    """情景记忆集合。"""

    episodes: list[Episode] = field(default_factory=list)


@dataclass(slots=True)
class MemoryEntry:
    """长期记忆条目。"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time_stamp: str = ""
    float_time_stamp: float = 0.0
    weekday: str = ""
    topic_id: int = 0
    topic_summary: str = ""
    memory: str = ""
    original_memory: str = ""
    compressed_memory: str = ""
    entry_type: str = "factual"
    speaker_id: str = "unknown"
    speaker_name: str = "User"
    consolidated: bool = False
    update_queue: list[Any] = field(default_factory=list)

    @property
    def memory_id(self) -> str:
        """Backward-compatible alias used by older InfoBudget code."""
        return self.id

    @memory_id.setter
    def memory_id(self, value: str) -> None:
        self.id = value

    @property
    def summary(self) -> str:
        """Backward-compatible retrieval text alias."""
        return self.memory

    @property
    def segment_id(self) -> str:
        """Best-effort segment id reconstructed from LightMem topic_id."""
        return f"seg_{self.topic_id + 1:06d}"

    @property
    def topic(self) -> str:
        """Backward-compatible topic label."""
        return str(self.topic_id)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。"""
        return asdict(self)


@dataclass(slots=True)
class CostLogEntry:
    """模型调用成本日志。"""

    call_id: str
    segment_id: str
    tier: Tier
    model_name: str
    backend: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    extraction_mode: str = "joint"
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。"""
        return asdict(self)


@dataclass(slots=True)
class RetrievalTrace:
    """单个问题的检索轨迹。"""

    question_id: str
    sample_id: str
    retrieved_memory_ids: list[str]
    retrieved_segment_ids: list[str]
    retrieved_summaries: list[str]
    evidence_session_ids: list[str] = field(default_factory=list)
    evidence_turn_refs: list[str] = field(default_factory=list)
    evidence_hit: bool = False
    evidence_recall_at_k: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。"""
        return asdict(self)


@dataclass(slots=True)
class ModelSpec:
    """模型注册信息。"""

    deploy: str
    backend: str
    model_name: str
    tokenizer_name: str
    max_context_tokens: int
    tensor_parallel_size: int
    dtype: str
    api_base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    request_model_name: str = ""

    @property
    def effective_model_name(self) -> str:
        """Return the provider-facing model name used in OpenAI-compatible requests."""
        return self.request_model_name or self.model_name

    def resolved_api_key(self) -> str:
        """Resolve an API key from either an explicit value or an environment variable."""
        if self.api_key.startswith("${") and self.api_key.endswith("}"):
            return os.getenv(self.api_key[2:-1], "")
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env, "")
        return ""


@dataclass(slots=True)
class PriceSpec:
    """价格注册信息。"""

    official_price_in_per_1m: float
    official_price_out_per_1m: float
    currency: str = "USD"

"""功能：定义 InfoBudget 全局数据结构。
输入：模块间传递的结构化数据。
输出：标准化 dataclass 与辅助方法。
依赖：标准库。
作者：OpenAI Codex
"""

from __future__ import annotations

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

    memory_id: str
    segment_id: str
    topic: str
    summary: str
    semantic_memory: SemanticMemory
    episodic_memory: EpisodicMemory
    importance: float
    information_score: float
    router_level: Tier
    extraction_mode: str
    extractor_name: str
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    created_at: str = field(default_factory=utc_now_iso)
    embedding_id: int = -1

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
class ModelSpec:
    """模型注册信息。"""

    deploy: str
    backend: str
    model_name: str
    tokenizer_name: str
    max_context_tokens: int
    tensor_parallel_size: int
    dtype: str


@dataclass(slots=True)
class PriceSpec:
    """价格注册信息。"""

    official_price_in_per_1m: float
    official_price_out_per_1m: float
    currency: str = "USD"

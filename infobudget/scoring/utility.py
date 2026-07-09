"""功能：实现记忆效用指标与未来扩展接口。
输入：Segment 与 MemoryStore。
输出：0 到 1 的新颖性分数。
依赖：abc、numpy、schemas、text、embeddings。
作者：OpenAI Codex
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
import re

import numpy as np

from infobudget.schemas import MemoryEntry, Segment
from infobudget.utils.embeddings import HashingTextEncoder, cosine_similarity
from infobudget.utils.text import clamp01, extract_entities, extract_event_clauses

ACTION_TERMS = {
    "add",
    "calculate",
    "call",
    "choose",
    "compare",
    "compute",
    "control",
    "define",
    "delete",
    "evaluate",
    "extract",
    "filter",
    "generate",
    "implement",
    "merge",
    "normalize",
    "output",
    "record",
    "retrieve",
    "route",
    "save",
    "select",
    "split",
    "update",
    "use",
    "采用",
    "保存",
    "比较",
    "调用",
    "定义",
    "更新",
    "记录",
    "计算",
    "路由",
    "选择",
    "输出",
    "修改",
    "删除",
    "实现",
    "抽取",
    "检索",
    "评估",
    "过滤",
    "合并",
    "拆分",
    "归一化",
    "控制",
    "降低",
    "提高",
}

CONDITION_TERMS = {
    "if",
    "when",
    "unless",
    "once",
    "whenever",
    "如果",
    "当",
    "若",
    "只要",
    "除非",
    "超过",
    "低于",
    "大于",
    "小于",
    "等于",
    "时",
    "情况下",
}

CONSTRAINT_TERMS = {
    "must",
    "should",
    "only",
    "never",
    "required",
    "require",
    "forbid",
    "prohibit",
    "need",
    "必须",
    "应该",
    "应当",
    "需要",
    "只",
    "只能",
    "不要",
    "不得",
    "禁止",
    "要求",
}

DECISION_TERMS = {
    "choose",
    "decision",
    "decide",
    "route",
    "routing",
    "select",
    "threshold",
    "priority",
    "prefer",
    "use",
    "选择",
    "决策",
    "决定",
    "路由",
    "阈值",
    "优先",
    "采用",
    "使用",
    "模型",
    "策略",
}

TARGET_TERMS = {
    "format",
    "json",
    "metric",
    "model",
    "output",
    "result",
    "score",
    "table",
    "字段",
    "格式",
    "结果",
    "输出",
    "模型",
    "指标",
    "表",
}

THRESHOLD_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_./+-]*\s*)?(?:>=|<=|>|<|=)\s*\d+(?:\.\d+)?|"
    r"\d+(?:\.\d+)?\s*(?:%|ms|s|minutes?|hours?|天|秒|分钟|小时)"
)
CODE_TOKEN_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_./+-]{2,}\b")


class MemorySearchable(Protocol):
    """支持向量检索的最小接口。"""

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        """按向量检索记忆。"""

    def is_empty(self) -> bool:
        """判断记忆库是否为空。"""


class BaseUtilityMetric(ABC):
    """效用指标基类。"""

    @abstractmethod
    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        """计算指标值。"""


class SemanticNoveltyScorer(BaseUtilityMetric):
    """语义新颖性。"""

    def __init__(self, encoder: HashingTextEncoder, top_k: int = 5):
        self.encoder = encoder
        self.top_k = top_k

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        if memory_store is None or memory_store.is_empty():
            return 1.0
        embedding = self.encoder.encode_text(segment.text)
        hits = memory_store.search_by_embedding(embedding, self.top_k)
        if not hits:
            return 1.0
        return clamp01(1.0 - max(score for _entry, score in hits))


class EntityNoveltyScorer(BaseUtilityMetric):
    """实体新颖性。"""

    def __init__(self, top_k: int = 5, encoder: HashingTextEncoder | None = None):
        self.top_k = top_k
        self.encoder = encoder

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        entities = {item.lower() for item in extract_entities(segment.text)}
        if memory_store is None or memory_store.is_empty():
            return 1.0 if not entities else 1.0
        if not entities:
            return 0.0
        query = self.encoder.encode_text(segment.text) if self.encoder else np.zeros(1, dtype=np.float32)
        hits = memory_store.search_by_embedding(query, self.top_k)
        known = {
            entity.lower()
            for entry, _score in hits
            for entity in extract_entities(entry.memory)
        }
        return clamp01(len(entities - known) / len(entities))


class EpisodicNoveltyScorer(BaseUtilityMetric):
    """情景新颖性。"""

    def __init__(self, encoder: HashingTextEncoder, top_k: int = 5):
        self.encoder = encoder
        self.top_k = top_k

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        events = extract_event_clauses(segment.text)
        if not events:
            return 0.0
        if memory_store is None or memory_store.is_empty():
            return 1.0
        dummy_query = self.encoder.encode_text(segment.text)
        hits = memory_store.search_by_embedding(dummy_query, self.top_k)
        existing = [self.encoder.encode_text(entry.memory) for entry, _score in hits]
        if not existing:
            return 1.0
        novelty_scores: list[float] = []
        for event in events:
            event_vec = self.encoder.encode_text(event)
            novelty_scores.append(1.0 - max(cosine_similarity(event_vec, seen) for seen in existing))
        return clamp01(float(np.mean(novelty_scores)))


def _contains_term(text: str, lowered: str, term: str) -> bool:
    if term.isascii():
        return re.search(rf"\b{re.escape(term.lower())}\b", lowered) is not None
    return term in text


def _contains_any(text: str, lowered: str, terms: set[str]) -> bool:
    return any(_contains_term(text, lowered, term) for term in terms)


def _has_threshold(text: str) -> bool:
    return THRESHOLD_PATTERN.search(text) is not None


def _has_object_anchor(text: str, lowered: str) -> bool:
    if extract_entities(text):
        return True
    if len(CODE_TOKEN_PATTERN.findall(text)) >= 2:
        return True
    return _contains_any(text, lowered, TARGET_TERMS)


def _weak_modality_factor(text: str, lowered: str) -> float:
    weak_terms = {"could", "maybe", "may", "might", "try", "consider", "可以考虑", "可能", "尝试", "考虑"}
    return 0.6 if _contains_any(text, lowered, weak_terms) else 1.0


class ActionabilityScorer(BaseUtilityMetric):
    """Score whether a segment can guide future decisions or executable behavior."""

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        text = segment.text or ""
        lowered = text.lower()
        has_action = _contains_any(text, lowered, ACTION_TERMS)
        has_object = _has_object_anchor(text, lowered)
        has_condition_word = _contains_any(text, lowered, CONDITION_TERMS)
        has_threshold = _has_threshold(text)
        has_constraint = _contains_any(text, lowered, CONSTRAINT_TERMS)
        has_decision = _contains_any(text, lowered, DECISION_TERMS)
        has_target = _contains_any(text, lowered, TARGET_TERMS)

        if not has_action:
            frame_score = 0.0
        elif not has_object:
            frame_score = 0.3
        elif has_condition_word or has_constraint:
            frame_score = 1.0
        elif has_target:
            frame_score = 0.8
        else:
            frame_score = 0.6

        if has_condition_word and has_threshold:
            condition_score = 1.0
        elif has_threshold:
            condition_score = 0.8
        elif has_condition_word:
            condition_score = 0.5
        else:
            condition_score = 0.0

        if has_constraint and (has_object or has_target):
            constraint_score = 1.0
        elif has_constraint:
            constraint_score = 0.7
        else:
            constraint_score = 0.0

        if has_decision and (has_condition_word or has_threshold):
            decision_score = 1.0
        elif has_decision:
            decision_score = 0.7
        else:
            decision_score = 0.0

        raw_score = max(frame_score, condition_score, constraint_score, decision_score)
        return clamp01(raw_score * _weak_modality_factor(text, lowered))


class PredictionGainScorer(BaseUtilityMetric):
    """未来扩展接口。"""

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        raise NotImplementedError("Prediction Gain is deferred in InfoBudget v1.0")


class InformationGainScorer(BaseUtilityMetric):
    """Information gain as max semantic/entity/episodic memory novelty."""

    def __init__(self, encoder: HashingTextEncoder, top_k: int = 5):
        self.semantic = SemanticNoveltyScorer(encoder, top_k)
        self.entity = EntityNoveltyScorer(top_k, encoder)
        self.episodic = EpisodicNoveltyScorer(encoder, top_k)

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        return max(
            self.semantic.compute(segment, memory_store),
            self.entity.compute(segment, memory_store),
            self.episodic.compute(segment, memory_store),
        )

"""功能：实现记忆效用指标与未来扩展接口。
输入：Segment 与 MemoryStore。
输出：0 到 1 的新颖性分数。
依赖：abc、numpy、schemas、text、embeddings。
作者：OpenAI Codex
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from infobudget.schemas import MemoryEntry, Segment
from infobudget.utils.embeddings import HashingTextEncoder, cosine_similarity
from infobudget.utils.text import clamp01, extract_entities, extract_event_clauses


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

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        entities = {item.lower() for item in extract_entities(segment.text)}
        if memory_store is None or memory_store.is_empty():
            return 1.0 if not entities else 1.0
        if not entities:
            return 0.0
        dummy_query = np.zeros(1, dtype=np.float32)
        hits = memory_store.search_by_embedding(dummy_query, self.top_k)
        known = {
            entity.name.lower()
            for entry, _score in hits
            for entity in entry.semantic_memory.entities
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
        existing = [
            self.encoder.encode_text(
                " ".join(
                    [
                        episode.subject,
                        episode.verb,
                        episode.object,
                        episode.time,
                    ]
                )
            )
            for entry, _score in hits
            for episode in entry.episodic_memory.episodes
        ]
        if not existing:
            return 1.0
        novelty_scores: list[float] = []
        for event in events:
            event_vec = self.encoder.encode_text(event)
            novelty_scores.append(1.0 - max(cosine_similarity(event_vec, seen) for seen in existing))
        return clamp01(float(np.mean(novelty_scores)))


class ActionabilityScorer(BaseUtilityMetric):
    """未来扩展接口。"""

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        raise NotImplementedError("Actionability is deferred in InfoBudget v1.0")


class PredictionGainScorer(BaseUtilityMetric):
    """未来扩展接口。"""

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        raise NotImplementedError("Prediction Gain is deferred in InfoBudget v1.0")


class InformationGainScorer(BaseUtilityMetric):
    """未来扩展接口。"""

    def compute(self, segment: Segment, memory_store: MemorySearchable | None) -> float:
        raise NotImplementedError("Information Gain is deferred in InfoBudget v1.0")

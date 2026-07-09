"""功能：实现内在信息指标。
输入：文本。
输出：0 到 1 的指标分数。
依赖：abc、text。
作者：OpenAI Codex
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from infobudget.utils.text import (
    clamp01,
    content_tokens,
    extract_entities,
    extract_idea_units,
    normalized_entropy,
    tokenize_text,
)


class BaseIntrinsicMetric(ABC):
    """内在指标基类。"""

    @abstractmethod
    def compute(self, text: str) -> float:
        """计算指标值。"""


class EntropyScorer(BaseIntrinsicMetric):
    """归一化熵。"""

    def compute(self, text: str) -> float:
        return normalized_entropy(tokenize_text(text))


class LexicalDensityScorer(BaseIntrinsicMetric):
    """词汇密度。"""

    def compute(self, text: str) -> float:
        tokens = tokenize_text(text)
        if not tokens:
            return 0.0
        return clamp01(len(content_tokens(text)) / len(tokens))


class EntityDensityScorer(BaseIntrinsicMetric):
    """实体密度。"""

    def compute(self, text: str) -> float:
        tokens = tokenize_text(text)
        if not tokens:
            return 0.0
        return clamp01(len(extract_entities(text)) / len(tokens))


class ConceptDensityScorer(BaseIntrinsicMetric):
    """概念密度。"""

    def __init__(self, spacy_model: str = ""):
        self.spacy_model = spacy_model

    def compute(self, text: str) -> float:
        tokens = tokenize_text(text)
        if not tokens:
            return 0.0
        return clamp01(len(extract_idea_units(text, self.spacy_model)) / len(tokens))

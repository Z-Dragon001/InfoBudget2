"""功能：定义联合提取器接口。
输入：Segment、tier、ScoreResult。
输出：MemoryEntry。
依赖：abc、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from infobudget.schemas import MemoryEntry, ScoreResult, Segment, Tier


class BaseExtractor(ABC):
    """提取器基类。"""

    @abstractmethod
    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> list[MemoryEntry]:
        """执行记忆提取。"""


class JointMemoryExtractor(BaseExtractor):
    """联合提取器抽象基类。"""

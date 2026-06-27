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
    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> MemoryEntry:
        """执行记忆提取。"""


class JointMemoryExtractor(BaseExtractor):
    """联合提取器抽象基类。"""


class LocalJointExtractor(JointMemoryExtractor):
    """本地联合提取器占位类。"""

    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> MemoryEntry:
        raise NotImplementedError("LocalJointExtractor is deferred in InfoBudget v1.0")


class APIJointExtractor(JointMemoryExtractor):
    """API 联合提取器占位类。"""

    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> MemoryEntry:
        raise NotImplementedError("APIJointExtractor is deferred in InfoBudget v1.0")

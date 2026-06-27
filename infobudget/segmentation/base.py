"""功能：定义主题分段器抽象接口。
输入：Turn 列表。
输出：Segment 列表与可选 trace。
依赖：abc、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from infobudget.schemas import Segment, Turn


class BaseSegmenter(ABC):
    """主题分段器基类。"""

    @abstractmethod
    def segment(self, turns: list[Turn]) -> list[Segment]:
        """执行主题分段。"""

    @abstractmethod
    def segment_with_trace(self, turns: list[Turn]) -> tuple[list[Segment], dict]:
        """执行主题分段并返回 trace。"""

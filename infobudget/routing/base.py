"""功能：定义预算路由器抽象接口。
输入：分数或分数列表。
输出：tier 或未实现异常。
依赖：abc、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from infobudget.schemas import Tier


class BaseRouter(ABC):
    """预算路由器基类。"""

    @abstractmethod
    def route(self, score: float) -> Tier:
        """路由单个分数。"""

    @abstractmethod
    def route_batch(self, scores: list[float]) -> list[Tier]:
        """路由一批分数。"""

    @abstractmethod
    def fit_percentiles(self, scores: list[float]) -> None:
        """拟合分位点。"""

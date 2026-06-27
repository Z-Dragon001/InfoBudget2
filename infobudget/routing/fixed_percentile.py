"""功能：实现 P33/P67 固定分位路由。
输入：score。
输出：small、medium 或 large。
依赖：schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass

from infobudget.routing.base import BaseRouter
from infobudget.schemas import Tier


@dataclass(slots=True)
class BudgetAwareRouter(BaseRouter):
    """InfoBudget v1.0 默认路由器。"""

    p33: float
    p67: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p33 < self.p67 <= 1.0:
            raise ValueError("router thresholds must satisfy 0 <= p33 < p67 <= 1")

    def route(self, score: float) -> Tier:
        if score < self.p33:
            return "small"
        if score < self.p67:
            return "medium"
        return "large"

    def route_batch(self, scores: list[float]) -> list[Tier]:
        return [self.route(score) for score in scores]

    def fit_percentiles(self, scores: list[float]) -> None:
        raise NotImplementedError("Deferred in InfoBudget v1.0")

"""功能：实现基础评估指标与成本汇总。
输入：预测结果、成本日志与路由分布。
输出：EvaluationMetrics。
依赖：dataclasses、collections。
作者：OpenAI Codex
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from infobudget.schemas import CostLogEntry, Tier


@dataclass(slots=True)
class EvaluationMetrics:
    """基础评估指标。"""

    accuracy: float
    precision: float
    recall: float
    total_cost_usd: float
    avg_cost_per_query: float
    avg_cost_per_memory: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    api_calls: int
    local_calls: int
    build_latency_ms: int
    qa_latency_ms: int
    router_distribution: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_accuracy(labels: list[bool]) -> float:
    """计算 Accuracy。"""
    return sum(1 for item in labels if item) / len(labels) if labels else 0.0


def compute_precision_recall(labels: list[bool]) -> tuple[float, float]:
    """在正类定义为回答正确时计算 Precision / Recall。"""
    if not labels:
        return 0.0, 0.0
    tp = sum(1 for item in labels if item)
    fp = len(labels) - tp
    fn = len(labels) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def aggregate_metrics(
    *,
    correctness: list[bool],
    cost_logs: list[CostLogEntry],
    routed_tiers: list[Tier],
    num_queries: int,
    num_memories: int,
    qa_latency_ms: int = 0,
) -> EvaluationMetrics:
    """聚合基础实验指标。"""
    accuracy = compute_accuracy(correctness)
    precision, recall = compute_precision_recall(correctness)
    input_tokens = sum(item.input_tokens for item in cost_logs)
    output_tokens = sum(item.output_tokens for item in cost_logs)
    total_cost = round(sum(item.cost_usd for item in cost_logs), 8)
    api_calls = sum(1 for item in cost_logs if item.backend == "api")
    local_calls = len(cost_logs) - api_calls
    latency = sum(item.latency_ms for item in cost_logs)
    counts = Counter(routed_tiers)
    denom = len(routed_tiers) or 1
    distribution = {
        "small": counts.get("small", 0) / denom,
        "medium": counts.get("medium", 0) / denom,
        "large": counts.get("large", 0) / denom,
    }
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        total_cost_usd=total_cost,
        avg_cost_per_query=total_cost / num_queries if num_queries else 0.0,
        avg_cost_per_memory=total_cost / num_memories if num_memories else 0.0,
        total_tokens=input_tokens + output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_calls=api_calls,
        local_calls=local_calls,
        build_latency_ms=latency,
        qa_latency_ms=qa_latency_ms,
        router_distribution=distribution,
    )


def pareto_front(runs: list[dict]) -> list[dict]:
    """计算 cost-accuracy Pareto front。"""
    front: list[dict] = []
    for candidate in runs:
        dominated = False
        for other in runs:
            if other is candidate:
                continue
            if (
                other["total_cost_usd"] <= candidate["total_cost_usd"]
                and other["accuracy"] >= candidate["accuracy"]
                and (
                    other["total_cost_usd"] < candidate["total_cost_usd"]
                    or other["accuracy"] > candidate["accuracy"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: (item["total_cost_usd"], -item["accuracy"]))

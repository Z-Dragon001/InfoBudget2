"""A/B/C artifact aggregation and scale-free counterfactual validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def aggregate_segment_usage(trace_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate question-level retrieval trace A into segment usage artifact B."""
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in trace_rows:
        scope = (
            str(row.get("dataset") or row.get("dataset_name") or ""),
            str(row.get("split") or ""),
            str(row.get("sample_id") or ""),
        )
        question_id = str(row.get("question_id") or "")
        if not all(scope) or not question_id:
            raise ValueError("QA trace row is missing dataset/split/sample_id/question_id")
        segment_ids = sorted(
            {
                str(item.get("segment_id") or "")
                for item in row.get("retrieved", ())
                if item.get("segment_id")
            }
        )
        for segment_id in segment_ids:
            key = (*scope, segment_id)
            record = grouped.setdefault(
                key,
                {
                    "dataset": scope[0],
                    "split": scope[1],
                    "sample_id": scope[2],
                    "segment_id": segment_id,
                    "question_ids": [],
                    "correct_count": 0,
                    "categories": defaultdict(int),
                },
            )
            record["question_ids"].append(question_id)
            record["correct_count"] += int(bool(row.get("correct")))
            record["categories"][str(row.get("category") or "unknown")] += 1
    output = []
    for key in sorted(grouped):
        record = grouped[key]
        count = len(record["question_ids"])
        output.append(
            {
                **{name: value for name, value in record.items() if name not in {"categories", "correct_count"}},
                "question_count": count,
                "accuracy": record["correct_count"] / count if count else 0.0,
                "category_counts": dict(sorted(record["categories"].items())),
            }
        )
    return output


def counterfactual_consistency(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Compare local-quality deltas with QA deltas without subtracting their scales."""
    pairs: list[tuple[float, float]] = []
    for row in rows:
        local_delta = float(row["predicted_quality_delta"])
        qa_delta = float(row["qa_delta"])
        pairs.append((local_delta, qa_delta))
    if not pairs:
        return {"count": 0, "sign_agreement": 0.0, "spearman": 0.0}
    non_ties = [(left, right) for left, right in pairs if left != 0.0 and right != 0.0]
    sign_agreement = (
        sum((left > 0) == (right > 0) for left, right in non_ties) / len(non_ties)
        if non_ties
        else 0.0
    )
    left_ranks = _average_ranks([left for left, _ in pairs])
    right_ranks = _average_ranks([right for _, right in pairs])
    return {
        "count": len(pairs),
        "non_tie_count": len(non_ties),
        "sign_agreement": sign_agreement,
        "spearman": _pearson(left_ranks, right_ranks),
    }


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_scale = sum((y - right_mean) ** 2 for y in right) ** 0.5
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0

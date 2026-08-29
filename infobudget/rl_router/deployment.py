"""Audit helpers for frozen-router deployment extraction and evaluation."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict
from typing import Any

from infobudget.rl_router.candidates import CandidateGenerationSummary
from infobudget.rl_router.schemas import TIERS, Tier, TopicSegment


def validate_route_decisions(
    segments: list[TopicSegment], actions: list[Tier]
) -> dict[str, Tier]:
    """Return an ordered route map after enforcing one action per unique segment."""
    if not segments:
        raise ValueError("deployment evaluation requires at least one segment")
    if len(segments) != len(actions):
        raise ValueError("deployment route must contain exactly one action per segment")
    ids = [segment.segment_id for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError("deployment route contains duplicate segment_id values")
    invalid = [tier for tier in actions if tier not in TIERS]
    if invalid:
        raise ValueError(f"deployment route contains invalid tiers: {invalid}")
    return dict(zip(ids, actions))


def deployment_namespace(
    source_namespace: str, *, protocol: str, fold: int, deployment_run_id: str
) -> str:
    """Derive an isolated, deterministic Qdrant namespace for one fold run."""
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", deployment_run_id).strip("_")[:24]
    digest = hashlib.sha256(
        f"{source_namespace}|{protocol}|{fold}|{deployment_run_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{source_namespace}_deploy_f{fold}_{readable}_{digest}"


def summarize_deployment_costs(
    summaries: list[CandidateGenerationSummary], *, question_count: int, currency: str = "USD"
) -> dict[str, Any]:
    """Aggregate extraction usage with both sample and QA denominators."""
    sample_count = len(summaries)
    totals = {
        "known_cost": sum(item.known_cost for item in summaries),
        "unknown_cost_attempts": sum(item.unknown_cost_attempts for item in summaries),
        "logical_api_calls": sum(
            int(item.attempt_summary.get("logical_api_calls", 0)) for item in summaries
        ),
        "successful_attempts": sum(
            int(item.attempt_summary.get("successful_attempts", 0)) for item in summaries
        ),
        "failed_attempts": sum(
            int(item.attempt_summary.get("failed_attempts", 0)) for item in summaries
        ),
        "repair_calls": sum(
            int(item.attempt_summary.get("repair_calls", 0)) for item in summaries
        ),
        "input_tokens": sum(
            int(item.attempt_summary.get("provider_input_tokens", 0)) for item in summaries
        ),
        "output_tokens": sum(
            int(item.attempt_summary.get("provider_output_tokens", 0)) for item in summaries
        ),
    }
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]

    def per(value: float | int, denominator: int) -> float | None:
        return float(value) / denominator if denominator else None

    calls = int(totals["logical_api_calls"])
    by_tier = {}
    for tier in TIERS:
        values = [
            (item.attempt_summary.get("by_tier") or {}).get(tier, {})
            for item in summaries
        ]
        tier_calls = sum(int(item.get("logical_api_calls", 0)) for item in values)
        tier_input = sum(int(item.get("input_tokens", 0)) for item in values)
        tier_output = sum(int(item.get("output_tokens", 0)) for item in values)
        by_tier[tier] = {
            "logical_api_calls": tier_calls,
            "input_tokens": tier_input,
            "output_tokens": tier_output,
            "total_tokens": tier_input + tier_output,
            "known_cost": sum(float(item.get("known_cost", 0.0)) for item in values),
            "unknown_cost_attempts": sum(
                int(item.get("unknown_cost_attempts", 0)) for item in values
            ),
            "average_input_tokens_per_call": per(tier_input, tier_calls),
            "average_output_tokens_per_call": per(tier_output, tier_calls),
        }
    return {
        "currency": currency,
        "objective_cost_scope": "memory_extraction_only",
        "sample_count": sample_count,
        "question_count": question_count,
        "totals": totals,
        "by_tier": by_tier,
        "per_sample": {
            "cost": per(totals["known_cost"], sample_count),
            "api_calls": per(calls, sample_count),
            "input_tokens": per(totals["input_tokens"], sample_count),
            "output_tokens": per(totals["output_tokens"], sample_count),
            "total_tokens": per(totals["total_tokens"], sample_count),
        },
        "per_question": {
            "amortized_extraction_cost": per(totals["known_cost"], question_count),
            "amortized_api_calls": per(calls, question_count),
            "amortized_input_tokens": per(totals["input_tokens"], question_count),
            "amortized_output_tokens": per(totals["output_tokens"], question_count),
            "amortized_total_tokens": per(totals["total_tokens"], question_count),
        },
        "per_extraction_call": {
            "input_tokens": per(totals["input_tokens"], calls),
            "output_tokens": per(totals["output_tokens"], calls),
            "total_tokens": per(totals["total_tokens"], calls),
        },
        "samples": [asdict(item) for item in summaries],
    }


def summarize_qa_usage(evaluations: list[Any]) -> dict[str, Any]:
    """Aggregate Reader/Judge token and cost fields from QA evaluations."""
    result: dict[str, Any] = {"question_count": len(evaluations)}
    for role in ("reader", "judge"):
        input_tokens = sum(
            int(_item_value(item, f"{role}_input_tokens", 0))
            for item in evaluations
        )
        output_tokens = sum(
            int(_item_value(item, f"{role}_output_tokens", 0))
            for item in evaluations
        )
        input_cost = sum(
            float(_item_value(item, f"{role}_input_cost", 0.0))
            for item in evaluations
        )
        output_cost = sum(
            float(_item_value(item, f"{role}_output_cost", 0.0))
            for item in evaluations
        )
        retry_count = sum(
            int(_item_value(item, f"{role}_retry_count", 0))
            for item in evaluations
        )
        result[role] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "logical_api_calls": len(evaluations),
            "reported_retry_count": retry_count,
            "transport_attempts": len(evaluations) + retry_count,
        }
    result["total_cost"] = result["reader"]["total_cost"] + result["judge"]["total_cost"]
    result["total_tokens"] = result["reader"]["total_tokens"] + result["judge"]["total_tokens"]
    return result


def build_question_outcomes(
    questions: list[dict[str, Any]], evaluations: list[Any]
) -> list[dict[str, Any]]:
    """Join immutable question metadata to Judge outcomes by question_id."""
    question_by_id = {str(item.get("question_id") or ""): item for item in questions}
    if "" in question_by_id or len(question_by_id) != len(questions):
        raise ValueError("evaluation questions require unique, non-empty question_id values")
    outcome_by_id: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        question_id = str(_item_value(evaluation, "question_id", ""))
        if question_id not in question_by_id:
            raise ValueError(f"evaluation returned unknown question_id: {question_id!r}")
        if question_id in outcome_by_id:
            raise ValueError(f"evaluation returned duplicate question_id: {question_id!r}")
        question = question_by_id[question_id]
        outcome_by_id[question_id] = {
            "question_id": question_id,
            "category": str(question.get("category") or "uncategorized"),
            "question_type": str(
                question.get("question_type") or "uncategorized"
            ),
            "judge_profile": str(question.get("judge_profile") or "generic"),
            "is_unanswerable": bool(question.get("is_unanswerable")),
            "correct": bool(_item_value(evaluation, "correct", False)),
        }
    missing = [item for item in question_by_id if item not in outcome_by_id]
    if missing:
        raise ValueError(
            f"evaluation outcomes are missing {len(missing)} question(s): {missing[:3]}"
        )
    return [outcome_by_id[str(item["question_id"])] for item in questions]


def summarize_question_outcomes(
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute overall and per-class binary Judge metrics with ddof=0."""
    overall = _judge_group_summary(outcomes, label="Overall")
    by_category = _summarize_groups(outcomes, "category", _category_label)
    by_question_type = _summarize_groups(
        outcomes, "question_type", lambda value: value
    )
    return {
        "std_definition": (
            "population standard deviation of binary judge_correct (ddof=0)"
        ),
        "overall": overall,
        "category_distribution": {
            key: {
                "label": value["label"],
                "count": value["count"],
                "fraction": value["fraction"],
            }
            for key, value in by_category.items()
        },
        "by_category": by_category,
        "question_type_distribution": {
            key: {
                "label": value["label"],
                "count": value["count"],
                "fraction": value["fraction"],
            }
            for key, value in by_question_type.items()
        },
        "by_question_type": by_question_type,
    }


def _summarize_groups(
    outcomes: list[dict[str, Any]], field: str, labeler
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        key = str(outcome.get(field) or "uncategorized")
        grouped.setdefault(key, []).append(outcome)
    total = len(outcomes)
    result = {}
    for key in sorted(grouped, key=_natural_group_key):
        summary = _judge_group_summary(grouped[key], label=labeler(key))
        summary["fraction"] = summary["count"] / total if total else 0.0
        result[key] = summary
    return result


def _judge_group_summary(
    outcomes: list[dict[str, Any]], *, label: str
) -> dict[str, Any]:
    count = len(outcomes)
    correct = sum(bool(item.get("correct")) for item in outcomes)
    mean = correct / count if count else 0.0
    std = math.sqrt(mean * (1.0 - mean)) if count else 0.0
    return {
        "label": label,
        "count": count,
        "correct_count": correct,
        "incorrect_count": count - correct,
        "accuracy": mean,
        "judge_correct": {"mean": mean, "std": std, "count": count},
    }


def _category_label(value: str) -> str:
    if value.startswith("category_"):
        return f"Category {value.removeprefix('category_')}"
    return "Uncategorized" if value == "uncategorized" else value


def _natural_group_key(value: str) -> tuple[str, int, int | str]:
    prefix, separator, suffix = value.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, 0, int(suffix)
    return value, 1, value


def _item_value(item: Any, field: str, default: Any) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)

"""Token-conserving cost allocation and deterministic deployment replay."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

from infobudget.rl_router.schemas import (
    ProviderUsage,
    ReplayResult,
    ReplaySegmentCost,
    SegmentAllocation,
    Tier,
    TopicSegment,
)
from infobudget.schemas import PriceSpec


def normalize_virtual_cost(value: float, all_small: float, all_large: float, epsilon: float = 1e-12) -> float:
    """Normalize deployment cost against the configured All-Small/All-Large baselines."""
    if all_large <= all_small:
        raise ValueError(
            f"All-Large cost must exceed All-Small cost; got {all_large} <= {all_small}"
        )
    return (float(value) - float(all_small)) / (float(all_large) - float(all_small) + epsilon)


def largest_remainder(total: int, weights: list[int | float]) -> list[int]:
    if total < 0 or any(weight < 0 for weight in weights):
        raise ValueError("totals and weights must be non-negative")
    if not weights:
        return []
    weight_sum = float(sum(weights))
    shares = [total / len(weights)] * len(weights) if weight_sum == 0 else [total * float(w) / weight_sum for w in weights]
    allocated = [math.floor(value) for value in shares]
    remainder = total - sum(allocated)
    order = sorted(range(len(weights)), key=lambda i: (-(shares[i] - allocated[i]), i))
    for index in order[:remainder]:
        allocated[index] += 1
    return allocated


def allocate_batch(
    usage: ProviderUsage,
    segment_ids: list[str],
    input_weights: list[int],
    output_weights: list[int],
    fact_counts: list[int],
    price: PriceSpec,
) -> list[SegmentAllocation]:
    if not (len(segment_ids) == len(input_weights) == len(output_weights) == len(fact_counts)):
        raise ValueError("allocation inputs must have equal lengths")
    input_tokens = largest_remainder(usage.input_tokens, input_weights)
    output_tokens = largest_remainder(usage.output_tokens, output_weights)
    result = []
    for index, segment_id in enumerate(segment_ids):
        result.append(
            SegmentAllocation(
                segment_id=segment_id,
                input_tokens=input_tokens[index],
                output_tokens=output_tokens[index],
                input_cost=input_tokens[index] * price.official_price_in_per_1m / 1_000_000,
                output_cost=output_tokens[index] * price.official_price_out_per_1m / 1_000_000,
                fact_count=fact_counts[index],
                serialized_input_tokens=input_weights[index],
                attributed_output_tokens=output_weights[index],
            )
        )
    return result


def allocate_fallback_recovery(
    parent_usage: ProviderUsage,
    singleton_usage_by_segment: dict[str, ProviderUsage],
    segment_ids: list[str],
    content_token_weights: list[int],
    fact_counts: list[int],
    price: PriceSpec,
) -> list[SegmentAllocation]:
    """Allocate failed-parent usage by content weight, then add each child exactly."""
    if not (
        len(segment_ids) == len(content_token_weights) == len(fact_counts)
    ):
        raise ValueError("fallback allocation inputs must have equal lengths")
    if set(singleton_usage_by_segment) != set(segment_ids):
        raise ValueError("fallback allocation requires one singleton usage per segment")
    parent_inputs = largest_remainder(
        parent_usage.input_tokens, content_token_weights
    )
    parent_outputs = largest_remainder(
        parent_usage.output_tokens, content_token_weights
    )
    weight_total = float(sum(content_token_weights))
    result: list[SegmentAllocation] = []
    for index, segment_id in enumerate(segment_ids):
        child = singleton_usage_by_segment[segment_id]
        weight = (
            float(content_token_weights[index]) / weight_total
            if weight_total
            else 1.0 / len(segment_ids)
        )
        input_tokens = parent_inputs[index] + child.input_tokens
        output_tokens = parent_outputs[index] + child.output_tokens
        result.append(
            SegmentAllocation(
                segment_id=segment_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=(
                    (
                        parent_usage.input_tokens * weight
                        + child.input_tokens
                    )
                    * price.official_price_in_per_1m
                    / 1_000_000
                ),
                output_cost=(
                    (
                        parent_usage.output_tokens * weight
                        + child.output_tokens
                    )
                    * price.official_price_out_per_1m
                    / 1_000_000
                ),
                fact_count=fact_counts[index],
                serialized_input_tokens=content_token_weights[index],
                attributed_output_tokens=child.output_tokens,
            )
        )
    return result


def replay_virtual_cost(
    segments: list[TopicSegment],
    actions: list[Tier],
    historical: dict[tuple[str, Tier], ReplaySegmentCost],
    buffer_config: dict[str, dict],
    prices: dict[Tier, PriceSpec],
    shared_prompt_tokens: dict[Tier, int],
) -> ReplayResult:
    if len(segments) != len(actions):
        raise ValueError("one action is required per segment")
    queues: dict[Tier, list[ReplaySegmentCost]] = {tier: [] for tier in prices}
    totals = defaultdict(float)
    batches = defaultdict(int)

    def flush(tier: Tier) -> None:
        queue = queues[tier]
        if not queue:
            return
        input_tokens = shared_prompt_tokens[tier] + sum(item.serialized_input_tokens for item in queue)
        output_tokens = sum(item.attributed_output_tokens for item in queue)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["input_cost"] += input_tokens * prices[tier].official_price_in_per_1m / 1_000_000
        totals["output_cost"] += output_tokens * prices[tier].official_price_out_per_1m / 1_000_000
        batches[tier] += 1
        queue.clear()

    for segment, tier in zip(segments, actions):
        item = historical.get((segment.segment_id, tier))
        if item is None:
            raise KeyError(f"missing historical token record for {segment.segment_id}/{tier}")
        cfg = buffer_config[tier]
        queue = queues[tier]
        proposed_input = shared_prompt_tokens[tier] + sum(x.serialized_input_tokens for x in queue) + item.serialized_input_tokens
        proposed_total = proposed_input + sum(x.attributed_output_tokens for x in queue) + item.attributed_output_tokens
        if queue and (len(queue) >= cfg["max_segments"] or proposed_input > cfg["max_input_tokens"] or proposed_total > cfg["max_total_context_tokens"]):
            flush(tier)
        queues[tier].append(item)
    for tier in queues:
        flush(tier)
    return ReplayResult(
        int(totals["input_tokens"]),
        int(totals["output_tokens"]),
        totals["input_cost"],
        totals["output_cost"],
        dict(batches),
    )

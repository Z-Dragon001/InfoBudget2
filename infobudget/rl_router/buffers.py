"""Independent deterministic extraction buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from infobudget.rl_router.parsing import render_extraction_prompt
from infobudget.rl_router.schemas import Tier, TopicSegment

TokenCounter = Callable[[str], int]
FlushHandler = Callable[[Tier, str, list[TopicSegment], str], None]


class OversizeSegmentError(ValueError):
    pass


@dataclass(slots=True)
class ExtractionBuffer:
    tier: Tier
    prompt: str
    max_segments: int
    max_input_tokens: int
    max_total_context_tokens: int
    reserve_output_tokens_per_segment: int
    token_counter: TokenCounter
    on_flush: FlushHandler
    allow_oversize_singleton: bool = False
    sample_id: str | None = None
    segments: list[TopicSegment] = field(default_factory=list)
    buffer_sequence: int = 0

    def add(self, segment: TopicSegment) -> None:
        if self.sample_id is not None and segment.sample_id != self.sample_id:
            raise ValueError("extraction buffers cannot mix samples")
        single_input = self._input_tokens([segment])
        single_total = single_input + self.reserve_output_tokens_per_segment
        if single_total > self.max_total_context_tokens:
            raise OversizeSegmentError(f"oversize_segment: {segment.segment_id} for tier {self.tier}")
        if single_input > self.max_input_tokens:
            if not self.allow_oversize_singleton:
                raise OversizeSegmentError(
                    f"oversize_segment: {segment.segment_id} for tier {self.tier}"
                )
            if self.segments:
                self.flush()
            self.sample_id = segment.sample_id
            self.segments.append(segment)
            self.flush()
            return
        if self.segments and self._would_overflow(segment):
            self.flush()
        self.sample_id = segment.sample_id
        self.segments.append(segment)

    def finalize(self) -> None:
        self.flush()

    def flush(self) -> None:
        if not self.segments:
            return
        assert self.sample_id is not None
        batch = list(self.segments)
        prompt = render_extraction_prompt(self.prompt, self.tier, batch)
        self.on_flush(self.tier, self.sample_id, batch, prompt)
        self.segments.clear()
        self.sample_id = None
        self.buffer_sequence += 1

    def _would_overflow(self, segment: TopicSegment) -> bool:
        proposed = [*self.segments, segment]
        input_tokens = self._input_tokens(proposed)
        return (
            len(self.segments) >= self.max_segments
            or input_tokens > self.max_input_tokens
            or input_tokens + len(proposed) * self.reserve_output_tokens_per_segment > self.max_total_context_tokens
        )

    def _input_tokens(self, segments: list[TopicSegment]) -> int:
        return self.token_counter(render_extraction_prompt(self.prompt, self.tier, segments))


def tier_config_int(config: dict, key: str, tier: Tier) -> int:
    """Read a tier-specific integer while accepting legacy scalar values."""
    value = config[key]
    if isinstance(value, dict):
        if tier not in value:
            raise ValueError(f"{key} is missing tier {tier}")
        value = value[tier]
    return int(value)


def build_tier_buffers(
    prompts: dict[Tier, str],
    config: dict,
    token_counters: dict[Tier, TokenCounter],
    handler: FlushHandler,
    tiers: tuple[Tier, ...] | list[Tier] | None = None,
) -> dict[Tier, ExtractionBuffer]:
    result = {}
    selected = tuple(tiers or ("small", "medium", "large"))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("extraction tiers must be non-empty and unique")
    for tier in selected:
        if tier not in prompts or tier not in token_counters:
            raise ValueError(f"missing prompt or token counter for tier {tier}")
        cfg = config["buffers"][tier]
        result[tier] = ExtractionBuffer(
            tier=tier,
            prompt=prompts[tier],
            max_segments=int(cfg["max_segments"]),
            max_input_tokens=int(cfg["max_input_tokens"]),
            max_total_context_tokens=int(cfg["max_total_context_tokens"]),
            reserve_output_tokens_per_segment=tier_config_int(
                config, "reserve_output_tokens_per_segment", tier
            ),
            token_counter=token_counters[tier],
            on_flush=handler,
            allow_oversize_singleton=bool(config.get("allow_oversize_singleton", False)),
        )
    return result

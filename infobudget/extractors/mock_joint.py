"""Mock joint memory extractor used for tests and local dry runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from infobudget.cost.logger import CostLogger
from infobudget.extractors.base import JointMemoryExtractor
from infobudget.runtime.model_registry import ModelRegistry
from infobudget.schemas import MemoryEntry, ScoreResult, Segment, Tier
from infobudget.utils.text import (
    count_tokens,
    summarize_text,
)


@dataclass(slots=True)
class MockJointExtractor(JointMemoryExtractor):
    """A heuristic extractor that mimics the I/O contract of an LLM extractor."""

    model_registry: ModelRegistry
    cost_logger: CostLogger
    prompt_template: str | dict[str, str]
    extraction_mode: str = "flat"

    def prompt_for_tier(self, tier: Tier) -> str:
        """Return the prompt template associated with the routed tier."""
        if isinstance(self.prompt_template, dict):
            return self.prompt_template.get(tier) or self.prompt_template.get("default") or next(
                iter(self.prompt_template.values())
            )
        return self.prompt_template

    @staticmethod
    def _render_prompt(template: str, *, tier: Tier, information_score: float, segment_text: str) -> str:
        """Render only the known placeholders without treating JSON braces as format slots."""
        escaped = (
            template.replace("{", "{{")
            .replace("}", "}}")
            .replace("{{router_level}}", "{router_level}")
            .replace("{{information_score}}", "{information_score}")
            .replace("{{segment_text}}", "{segment_text}")
        )
        return escaped.format(
            router_level=tier,
            information_score=f"{information_score:.4f}",
            segment_text=segment_text,
        )

    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> list[MemoryEntry]:
        model_spec = self.model_registry.get(tier)
        summary = summarize_text(segment.text, 120)
        filled_prompt = self._render_prompt(
            self.prompt_for_tier(tier),
            tier=tier,
            information_score=score_result.final_score,
            segment_text=segment.text,
        )
        input_tokens = count_tokens(filled_prompt)
        output_tokens = max(16, count_tokens(summary))
        latency_ms = max(80, 25 * count_tokens(segment.text))
        mode = self.extraction_mode.strip().lower()
        self.cost_logger.log_extraction(
            segment_id=segment.segment_id,
            tier=tier,
            model_spec=model_spec,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            extraction_mode="flat_factual" if mode == "flat" else "event_factual",
        )
        if mode == "event":
            self.cost_logger.log_extraction(
                segment_id=segment.segment_id,
                tier=tier,
                model_spec=model_spec,
                input_tokens=input_tokens,
                output_tokens=max(8, output_tokens // 2),
                latency_ms=latency_ms,
                extraction_mode="event_relational",
            )
        factual_entries = [_entry_from_line(line, segment, "factual") for line in segment.text.splitlines()]
        entries = [entry for entry in factual_entries if entry is not None]
        if mode == "event":
            relational_entries = [_entry_from_line(line, segment, "relational") for line in segment.text.splitlines()]
            entries.extend(entry for entry in relational_entries if entry is not None)
        return entries or [
            MemoryEntry(topic_id=_topic_id_from_segment(segment), memory=summary, original_memory=summary)
        ]


_SOURCE_LINE_RE = re.compile(
    r"^\[(?P<time>[^\],]+)(?:,\s*(?P<weekday>[^\]]+))?\]\s*"
    r"(?P<source_id>\d+)\.(?P<speaker>[^:]+):\s*(?P<content>.*)$"
)


def _entry_from_line(line: str, segment: Segment, entry_type: str) -> MemoryEntry | None:
    match = _SOURCE_LINE_RE.match(line.strip())
    if not match:
        return None
    content = match.group("content").strip()
    if not content:
        return None
    time_stamp = match.group("time").strip()
    speaker = match.group("speaker").strip()
    source_id = int(match.group("source_id"))
    if entry_type == "relational":
        memory = f"{speaker} engaged in the conversation about: {content}"
    else:
        memory = f"{speaker} said: {content}"
    return MemoryEntry(
        time_stamp=time_stamp,
        float_time_stamp=_float_timestamp(time_stamp),
        weekday=(match.group("weekday") or "").strip(),
        topic_id=_topic_id_from_segment(segment),
        memory=memory,
        original_memory=memory,
        entry_type=entry_type,
        speaker_id=speaker.lower() or "unknown",
        speaker_name=speaker or "User",
        source_segment_id=segment.segment_id,
        source_turn_id=source_id + 1,
        source_turn_ids=list(segment.turn_ids),
        source_start_turn=segment.start_turn,
        source_end_turn=segment.end_turn,
    )


def _topic_id_from_segment(segment: Segment) -> int:
    match = re.search(r"(\d+)$", segment.segment_id)
    return max(0, int(match.group(1)) - 1) if match else max(0, segment.start_turn - 1)


def _float_timestamp(time_stamp: str) -> float:
    if not time_stamp:
        return 0.0
    try:
        return datetime.fromisoformat(time_stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0

"""Read frozen segment and question artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from infobudget.rl_router.schemas import TopicSegment


def load_segments(path: str | Path) -> list[TopicSegment]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        segments = [TopicSegment.from_dict(json.loads(line)) for line in handle if line.strip()]
    if not segments:
        raise ValueError(f"no segments found in {source}")
    sample_id = segments[0].sample_id
    if any(segment.sample_id != sample_id for segment in segments):
        raise ValueError("segments.jsonl mixes samples")
    if len({segment.segment_id for segment in segments}) != len(segments):
        raise ValueError("duplicate segment_id")
    return sorted(segments, key=lambda item: (item.segment_order, item.start_turn))

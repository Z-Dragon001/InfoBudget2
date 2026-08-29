"""Input discovery and frozen JSONL materialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from infobudget.quality_router.io import write_jsonl
from infobudget.rl_router.schemas import TopicSegment


def load_topic_segments(path: str | Path) -> list[TopicSegment]:
    source = Path(path)
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(source.rglob("segments.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no segments.jsonl files found: {source}")
    result: list[TopicSegment] = []
    seen: set[str] = set()
    for item in paths:
        with item.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"segment row must be an object: {item}:{line_number}")
                segment = TopicSegment.from_dict(value)
                if segment.segment_id in seen:
                    raise ValueError(f"duplicate segment_id: {segment.segment_id}")
                seen.add(segment.segment_id)
                result.append(segment)
    return sorted(
        result,
        key=lambda item: (item.dataset_name, item.split, item.sample_id, item.segment_order),
    )


def export_reference_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    return write_jsonl(path, sorted(rows, key=_row_key))


def _row_key(row: dict[str, object]) -> tuple[str, str, str, int, str]:
    return (
        str(row.get("dataset_name") or row.get("dataset") or ""),
        str(row.get("split") or ""),
        str(row.get("sample_id") or ""),
        int(row.get("segment_order") or 0),
        str(row.get("segment_id") or ""),
    )

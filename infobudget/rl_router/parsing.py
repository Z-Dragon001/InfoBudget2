"""Strict renderer and parser for tiered JSON fact extraction."""

from __future__ import annotations

import json

from infobudget.rl_router.schemas import ParsedBatch, Tier, TopicSegment


def render_batch(segments: list[TopicSegment]) -> str:
    """Render batch topics with stable segment IDs around LightMem-style turn lines."""
    return "\n\n".join(
        f"--- Topic {segment.segment_id} ---\n{segment.text}" for segment in segments
    )


def render_extraction_prompt(template: str, tier: Tier, segments: list[TopicSegment]) -> str:
    """Replace only legacy placeholders so JSON braces remain untouched."""
    required = ("{router_level}", "{information_score}", "{segment_text}")
    missing = [placeholder for placeholder in required if placeholder not in template]
    if missing:
        raise ValueError(f"extraction prompt is missing placeholders: {missing}")
    return (
        template.replace("{router_level}", tier)
        .replace("{information_score}", "N/A (frozen candidate generation)")
        .replace("{segment_text}", render_batch(segments))
    )


def parse_fact_batch(
    content: str,
    expected_segment_ids: list[str],
    max_facts: int,
    expected_source_ids_by_segment: dict[str, set[int]] | None = None,
) -> ParsedBatch:
    """Parse JSON-only output and enforce segment, source, and per-tier fact limits."""
    try:
        root = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fact extraction response is not valid JSON: {exc.msg}") from exc
    if not isinstance(root, dict):
        raise ValueError("fact extraction JSON root must be an object")
    if set(root) != {"processed_segment_ids", "data"}:
        raise ValueError("fact extraction JSON must contain only processed_segment_ids and data")

    processed = root["processed_segment_ids"]
    if processed != expected_segment_ids:
        raise ValueError(
            "processed_segment_ids must exactly match input order: "
            f"expected {expected_segment_ids}, got {processed}"
        )
    data = root["data"]
    if not isinstance(data, list):
        raise ValueError("fact extraction data must be an array")

    expected = set(expected_segment_ids)
    facts_by_segment = {segment_id: [] for segment_id in expected_segment_ids}
    source_ids_by_segment: dict[str, list[list[int]]] = {
        segment_id: [] for segment_id in expected_segment_ids
    }
    serialized_items = {segment_id: [] for segment_id in expected_segment_ids}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"data[{index}] must be an object"
            )
        keys = set(item)
        legacy_keys = {"segment_id", "source_id", "fact"}
        multi_source_keys = {"segment_id", "source_ids", "fact"}
        if frozenset(keys) not in {frozenset(legacy_keys), frozenset(multi_source_keys)}:
            raise ValueError(
                f"data[{index}] must contain only segment_id, source_ids, and fact"
            )
        segment_id = item["segment_id"]
        if not isinstance(segment_id, str) or segment_id not in expected:
            raise ValueError(f"unknown segment_id in model output: {segment_id!r}")
        raw_source_ids = item.get("source_ids", [item.get("source_id")])
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            raise ValueError(f"source_ids for {segment_id} must be a non-empty array")
        source_ids: list[int] = []
        for source_id in raw_source_ids:
            if isinstance(source_id, bool) or not isinstance(source_id, int) or source_id < 0:
                raise ValueError(f"invalid source_id for {segment_id}: {source_id!r}")
            if source_id not in source_ids:
                source_ids.append(source_id)
        source_ids.sort()
        if expected_source_ids_by_segment is not None:
            allowed = expected_source_ids_by_segment[segment_id]
            invalid = [source_id for source_id in source_ids if source_id not in allowed]
            if invalid:
                raise ValueError(
                    f"source_id {invalid[0]} does not belong to segment {segment_id}"
                )
        fact = item["fact"]
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError(f"empty fact for {segment_id}")
        fact = fact.strip()
        if fact in facts_by_segment[segment_id]:
            existing_index = facts_by_segment[segment_id].index(fact)
            merged = sorted(
                set(source_ids_by_segment[segment_id][existing_index]) | set(source_ids)
            )
            source_ids_by_segment[segment_id][existing_index] = merged
            serialized_items[segment_id][existing_index]["source_ids"] = merged
            continue
        facts_by_segment[segment_id].append(fact)
        source_ids_by_segment[segment_id].append(source_ids)
        serialized_items[segment_id].append(
            {"segment_id": segment_id, "source_ids": source_ids, "fact": fact}
        )

    for segment_id, facts in facts_by_segment.items():
        if len(facts) > max_facts:
            raise ValueError(
                f"{segment_id} exceeds max_facts_per_segment={max_facts}"
            )
    blocks = {
        segment_id: json.dumps(
            {"processed_segment_ids": [segment_id], "data": serialized_items[segment_id]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for segment_id in expected_segment_ids
    }
    return ParsedBatch(facts_by_segment, source_ids_by_segment, blocks)


def is_schema_repairable(error: Exception) -> bool:
    """Only repair structure; semantic/source violations stay terminal."""
    message = str(error)
    non_repairable = (
        "unknown segment_id",
        "does not belong to segment",
        "exceeds max_facts_per_segment",
        "empty fact",
    )
    return not any(marker in message for marker in non_repairable)


def render_json_repair_prompt(
    *,
    invalid_output: str,
    validation_error: str,
    expected_segment_ids: list[str],
    expected_source_ids_by_segment: dict[str, set[int]],
    max_facts_per_segment: int,
) -> str:
    """Build a zero-temperature, schema-only repair request without hidden identifiers."""
    contract = {
        "processed_segment_ids": expected_segment_ids,
        "data": [
            {
                "segment_id": "one of processed_segment_ids",
                "source_ids": "a non-empty array of integers allowed for that segment",
                "fact": "a non-empty existing fact from the invalid output",
            }
        ],
    }
    allowed = {
        segment_id: sorted(source_ids)
        for segment_id, source_ids in expected_source_ids_by_segment.items()
    }
    return (
        "You are a deterministic JSON repair tool. Repair JSON structure only.\n"
        "Do not add new facts, rewrite facts, infer facts, or invent identifiers.\n"
        "Preserve every valid fact and all of its existing source_ids. Remove only malformed empty items.\n"
        f"Return no more than {max_facts_per_segment} facts for each topic segment.\n"
        "Output exactly one JSON object and no markdown or explanation.\n\n"
        f"Validation error:\n{validation_error}\n\n"
        "Required schema and exact processed order:\n"
        f"{json.dumps(contract, ensure_ascii=False)}\n\n"
        "Allowed source_id values by segment:\n"
        f"{json.dumps(allowed, ensure_ascii=False, sort_keys=True)}\n\n"
        "Invalid model output to repair:\n"
        f"{invalid_output}"
    )

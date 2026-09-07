from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reference_fact_pipeline.audit import audit_reference_facts


def _fact_id(segment_id: str, text: str, source_ids: list[int]) -> str:
    normalized = " ".join(text.casefold().strip().rstrip("。.!！?").split())
    payload = json.dumps(
        [segment_id, normalized, source_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "rf_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _segment(segment_id: str, order: int, turn_ids: list[int]) -> dict:
    rendered_turns = "\n".join(
        f"[2023-01-01T00:00:{index:02d}.000, Sun] {turn_id - 1}.Alice: Alice likes tea."
        for index, turn_id in enumerate(turn_ids)
    )
    return {
        "dataset_name": "locomo",
        "split": "full",
        "sample_id": "conv-1",
        "session_id": "session-1",
        "segment_id": segment_id,
        "segmentation_method": "nsp_text_tiling",
        "segmentation_version": "nsp_text_tiling_v1",
        "start_turn": min(turn_ids),
        "end_turn": max(turn_ids),
        "turn_ids": turn_ids,
        "start_timestamp": None,
        "end_timestamp": None,
        "text": rendered_turns,
        "token_count": 4,
        "source_content_hash": f"hash-{order}",
        "segment_order": order,
    }


def _reference_row(segment: dict, facts: list[dict]) -> dict:
    payload = [
        [fact["reference_fact_id"], fact["fact_text"], fact["source_turn_ids"]]
        for fact in facts
    ]
    return {
        "schema_version": "frozen_reference_fact_v1",
        "dataset_name": segment["dataset_name"],
        "dataset": segment["dataset_name"],
        "split": segment["split"],
        "sample_id": segment["sample_id"],
        "session_id": segment["session_id"],
        "segment_id": segment["segment_id"],
        "segment_order": segment["segment_order"],
        "segmentation_method": segment["segmentation_method"],
        "segmentation_version": segment["segmentation_version"],
        "source_content_hash": segment["source_content_hash"],
        "segment_turn_ids": segment["turn_ids"],
        "reference_set_hash": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "reference_facts": facts,
        "rejected_facts": [],
        "raw_proposal_count": len(facts),
        "grounded_accept_count": len(facts),
        "frozen_fact_count": len(facts),
        "truncated_to_k": False,
        "run_id": "test-run",
        "prompt_version": "test-prompt",
        "config_hash": "test-config",
    }


def test_audit_passes_structural_checks_and_queues_zero_fact_segment(tmp_path: Path) -> None:
    first = _segment("seg-1", 1, [1])
    second = _segment("seg-2", 2, [2])
    text = "Alice likes tea."
    fact = {
        "reference_fact_id": _fact_id(second["segment_id"], text, [2]),
        "fact_text": text,
        "text": text,
        "source_turn_ids": [2],
        "fact_type": "preference",
        "state_status": "timeless",
        "origin": "initial",
        "grounding_reason": "The source explicitly says this.",
        "selection_rank": 1,
    }
    segment_file = tmp_path / "segments" / "segments.jsonl"
    reference_file = tmp_path / "reference_facts.jsonl"
    manifest_file = tmp_path / "manifest.json"
    _write_jsonl(segment_file, [first, second])
    _write_jsonl(
        reference_file,
        [_reference_row(first, []), _reference_row(second, [fact])],
    )
    manifest_file.write_text(
        json.dumps(
            {
                "run_complete": True,
                "unresolved_failure_count": 0,
                "segment_count": 2,
                "fact_count": 1,
                "config_hash": "test-config",
                "run_id": "test-run",
                "cost_complete": True,
            }
        ),
        encoding="utf-8",
    )

    result = audit_reference_facts(
        references_path=reference_file,
        segments_path=segment_file,
        manifest_path=manifest_file,
        output_dir=tmp_path / "audit",
        random_sample_size=1,
    )

    assert result["status"] == "AUTOMATED_PASS_MANUAL_PENDING"
    assert result["error_count"] == 0
    assert result["zero_fact_segment_count"] == 1
    queue = [
        json.loads(line)
        for line in (tmp_path / "audit" / "manual_review_queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {item["category"] for item in queue} == {
        "zero_reference_set",
        "stratified_quality_sample",
    }
    zero_item = next(item for item in queue if item["category"] == "zero_reference_set")
    assert zero_item["source_id_convention"] == "canonical_one_based"
    assert zero_item["source_id_mapping"] == [
        {"legacy_dialogue_index": 0, "source_turn_id": 1}
    ]
    assert zero_item["segment_text"].startswith(
        "<SOURCE_TURN_ID=1> [2023-01-01T00:00:00.000, Sun] Alice:"
    )
    assert " 0.Alice:" not in zero_item["segment_text"]


def test_audit_rejects_invalid_source_turn(tmp_path: Path) -> None:
    segment = _segment("seg-1", 1, [1])
    text = "Alice likes tea."
    fact = {
        "reference_fact_id": _fact_id(segment["segment_id"], text, [99]),
        "fact_text": text,
        "text": text,
        "source_turn_ids": [99],
        "fact_type": "preference",
        "state_status": "timeless",
        "origin": "initial",
        "grounding_reason": "Incorrect source for the test.",
        "selection_rank": 1,
    }
    segment_file = tmp_path / "segments.jsonl"
    reference_file = tmp_path / "reference_facts.jsonl"
    manifest_file = tmp_path / "manifest.json"
    _write_jsonl(segment_file, [segment])
    _write_jsonl(reference_file, [_reference_row(segment, [fact])])
    manifest_file.write_text(
        json.dumps(
            {
                "run_complete": True,
                "unresolved_failure_count": 0,
                "segment_count": 1,
                "fact_count": 1,
                "config_hash": "test-config",
                "run_id": "test-run",
                "cost_complete": True,
            }
        ),
        encoding="utf-8",
    )

    result = audit_reference_facts(
        references_path=reference_file,
        segments_path=segment_file,
        manifest_path=manifest_file,
        output_dir=tmp_path / "audit",
        random_sample_size=0,
    )

    assert result["status"] == "AUTOMATED_FAIL"
    assert any(item["code"] == "invalid_source_turn_ids" for item in result["errors"])


def test_audit_renders_last_boundary_with_canonical_id(tmp_path: Path) -> None:
    segment = _segment("seg-boundary", 1, [360, 361, 370])
    text = "Melanie wanted blue streaks to convey tranquility."
    fact = {
        "reference_fact_id": _fact_id(segment["segment_id"], text, [370]),
        "fact_text": text,
        "text": text,
        "source_turn_ids": [370],
        "fact_type": "preference",
        "state_status": "current",
        "origin": "initial",
        "grounding_reason": "The final source turn explicitly states this.",
        "selection_rank": 1,
    }
    segment_file = tmp_path / "segments.jsonl"
    reference_file = tmp_path / "reference_facts.jsonl"
    manifest_file = tmp_path / "manifest.json"
    _write_jsonl(segment_file, [segment])
    _write_jsonl(reference_file, [_reference_row(segment, [fact])])
    manifest_file.write_text(
        json.dumps(
            {
                "run_complete": True,
                "unresolved_failure_count": 0,
                "segment_count": 1,
                "fact_count": 1,
                "config_hash": "test-config",
                "run_id": "test-run",
                "cost_complete": True,
            }
        ),
        encoding="utf-8",
    )

    result = audit_reference_facts(
        references_path=reference_file,
        segments_path=segment_file,
        manifest_path=manifest_file,
        output_dir=tmp_path / "audit",
        random_sample_size=1,
    )

    assert result["error_count"] == 0
    queue = [
        json.loads(line)
        for line in (tmp_path / "audit" / "manual_review_queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    item = queue[0]
    assert "<SOURCE_TURN_ID=370>" in item["segment_text"]
    assert " 369.Alice:" not in item["segment_text"]
    assert item["source_id_mapping"][-1] == {
        "legacy_dialogue_index": 369,
        "source_turn_id": 370,
    }

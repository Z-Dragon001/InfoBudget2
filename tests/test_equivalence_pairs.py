from __future__ import annotations

import json
from pathlib import Path

from infobudget.quality_router.equivalence_pairs import build_fact_equivalence_pairs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_fact_equivalence_pairs_filters_by_source_overlap(tmp_path: Path) -> None:
    segments = tmp_path / "segments.jsonl"
    references = tmp_path / "references.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "pairs.jsonl"
    manifest_path = tmp_path / "manifest.json"
    key = {
        "dataset_name": "dataset",
        "split": "full",
        "sample_id": "sample-1",
        "segment_id": "segment-1",
    }
    _write_jsonl(segments, [{**key, "turn_ids": [1, 2]}])
    _write_jsonl(
        references,
        [
            {
                **key,
                "reference_set_hash": "reference-hash",
                "reference_facts": [
                    {"reference_fact_id": "r1", "text": "Alice moved.", "source_turn_ids": [1]},
                    {"reference_fact_id": "r2", "text": "Bob stayed.", "source_turn_ids": [2]},
                ],
            }
        ],
    )
    _write_jsonl(
        candidates,
        [
            {
                **key,
                "model_id": "model-a",
                "fact_id": "c1",
                "text": "Alice relocated.",
                "source_turn_ids": [1],
                "candidate_extraction_run_id": "run-1",
            }
        ],
    )
    manifest = build_fact_equivalence_pairs(
        segments_path=segments,
        references_path=references,
        candidates_path=candidates,
        output_path=output,
        manifest_path=manifest_path,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["candidate_fact_id"] == "c1"
    assert rows[0]["reference_fact_id"] == "r1"
    assert rows[0]["shared_source_turn_ids"] == [1]
    assert manifest["raw_same_segment_pair_count"] == 2
    assert manifest["eligible_pair_count"] == 1
    assert manifest["excluded_no_source_overlap_count"] == 1

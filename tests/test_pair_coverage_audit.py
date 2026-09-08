from __future__ import annotations

import json
from pathlib import Path

from infobudget.quality_router.pair_coverage_audit import audit_fact_pair_coverage


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_reports_unique_coverage_and_risky_exclusions(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    references = tmp_path / "references.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    output = tmp_path / "audit.json"
    risky = tmp_path / "risky.jsonl"
    identity = {
        "dataset_name": "locomo",
        "split": "full",
        "sample_id": "conv-1",
        "segment_id": "seg-1",
    }
    _write_jsonl(
        candidates,
        [
            {**identity, "model_id": "m1", "fact_id": "c1", "text": "Alice moved.", "source_turn_ids": [1]},
            {**identity, "model_id": "m1", "fact_id": "c2", "text": "Bob stayed home.", "source_turn_ids": [2]},
        ],
    )
    _write_jsonl(
        references,
        [
            {
                **identity,
                "reference_facts": [
                    {"reference_fact_id": "r1", "text": "Alice relocated.", "source_turn_ids": [1]},
                    {"reference_fact_id": "r2", "text": "Bob stayed home.", "source_turn_ids": [3]},
                ],
            }
        ],
    )
    _write_jsonl(
        pairs,
        [
            {
                **identity,
                "pair_id": "p1",
                "model_id": "m1",
                "candidate_fact_id": "c1",
                "reference_fact_id": "r1",
            }
        ],
    )
    result = audit_fact_pair_coverage(
        candidates_path=candidates,
        references_path=references,
        pairs_path=pairs,
        output_path=output,
        risky_pairs_output_path=risky,
    )
    assert result["candidate_fact_count"] == 2
    assert result["reference_fact_count"] == 2
    assert result["raw_same_segment_pair_count"] == 4
    assert result["eligible_source_overlap_pair_count"] == 1
    assert result["unique_candidate_without_pair_count"] == 1
    assert result["reference_coverage_by_model"]["m1"]["without_eligible_pair"] == 1
    risky_rows = [json.loads(line) for line in risky.read_text(encoding="utf-8").splitlines()]
    assert any(row["candidate_fact_id"] == "c2" and row["reference_fact_id"] == "r2" for row in risky_rows)


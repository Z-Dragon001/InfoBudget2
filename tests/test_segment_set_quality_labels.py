from __future__ import annotations

import pytest

from scripts.build_fact_quality_labels import _score_result


def test_score_result_gives_partial_credit_but_applies_time_hard_gate() -> None:
    judgment = {
        "dataset_name": "locomo", "split": "full", "sample_id": "conv-1",
        "segment_id": "seg-1",
    }
    result = {
        "set_id": "A", "model_id": "model-a",
        "candidate_assessments": [
            {"candidate_id": "c1", "semantic_status": "SUPPORTED"},
            {"candidate_id": "c2", "semantic_status": "PARTIALLY_SUPPORTED"},
        ],
        "gold_fact_assessments": [
            {
                "gold_fact_id": "g1", "coverage_status": "FULL",
                "time_status": "MISSING_REQUIRED_EXACT_TIME",
                "covering_candidate_ids": ["c1"],
            },
            {
                "gold_fact_id": "g2", "coverage_status": "PARTIAL",
                "time_status": "NOT_APPLICABLE",
                "covering_candidate_ids": ["c2"],
            },
        ],
    }
    label, _ = _score_result(
        judgment=judgment, result=result,
        gold_fact_ids={"g1", "g2"}, profile_id="profile-a",
        candidate_extraction_run_id="run-a", reference_set_hash="hash-a",
    )
    assert label["strict_candidate_precision"] == 0.5
    assert label["partial_credit_candidate_precision"] == 0.75
    assert label["strict_gold_fact_recall"] == 0.0
    assert label["partial_credit_gold_coverage"] == 0.25
    assert label["temporal_recall"] == 0.0
    assert label["temporal_gold_fact_count"] == 1
    assert label["time_applicability_decided_by"] == "judge"
    assert label["set_quality_f2"] == pytest.approx(0.2884615385)

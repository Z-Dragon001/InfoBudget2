from __future__ import annotations

from scripts.build_fact_quality_labels import _score_result


def test_score_result_applies_temporal_hard_gate_and_f2() -> None:
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
        "gold_claim_assessments": [
            {
                "gold_fact_id": "g1", "claim_id": "g1:C1",
                "content_status": "COVERED", "time_status": "MISSING_REQUIRED_EXACT_TIME",
                "covering_candidate_ids": ["c1"],
            },
            {
                "gold_fact_id": "g1", "claim_id": "g1:C2",
                "content_status": "COVERED", "time_status": "NOT_APPLICABLE",
                "covering_candidate_ids": ["c2"],
            },
        ],
    }
    label, _ = _score_result(
        judgment=judgment, result=result,
        claims_by_fact={"g1": ["g1:C1", "g1:C2"]},
        required_time={"g1:C1": True, "g1:C2": False},
        profile_id="profile-a", candidate_extraction_run_id="run-a",
        reference_set_hash="hash-a",
    )
    assert label["strict_candidate_precision"] == 0.5
    assert label["strict_claim_recall"] == 0.5
    assert label["strict_gold_fact_recall"] == 0.0
    assert label["temporal_recall"] == 0.0
    assert label["set_quality_f2"] == 0.5

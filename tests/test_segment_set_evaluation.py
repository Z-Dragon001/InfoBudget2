from __future__ import annotations

import json

import pytest

from infobudget.quality_router.segment_set_evaluation import (
    _take_stratified,
    parse_gold_evaluation_units,
    parse_segment_set_judgment,
)


def _reference_row() -> dict:
    return {
        "dataset_name": "locomo",
        "split": "full",
        "sample_id": "conv-1",
        "segment_id": "seg-1",
        "reference_facts": [
            {
                "reference_fact_id": "g1",
                "text": "Alice moved on May 20, 2023 and started painting.",
            }
        ],
    }


def _task() -> dict:
    return {
        "segment_id": "seg-1",
        "dataset_name": "locomo",
        "split": "full",
        "sample_id": "conv-1",
        "model_by_set": {"A": "model-a"},
        "model_input": {
            "segment_id": "seg-1",
            "segment_text": "Alice moved on May 20, 2023 and started painting.",
            "gold_facts": [
                {
                    "gold_fact_id": "g1",
                    "original_text": "Alice moved on May 20, 2023 and started painting.",
                    "claim_units": [
                        {
                            "claim_id": "g1:C1",
                            "claim_text": "Alice moved.",
                            "required_time": {
                                "required": True,
                                "normalized_value": "2023-05-20",
                                "resolution": "day",
                                "surface_form": "May 20, 2023",
                            },
                        },
                        {
                            "claim_id": "g1:C2",
                            "claim_text": "Alice started painting.",
                            "required_time": {
                                "required": False,
                                "normalized_value": None,
                                "resolution": None,
                                "surface_form": None,
                            },
                        },
                    ],
                }
            ],
            "candidate_sets": [
                {
                    "set_id": "A",
                    "facts": [{"candidate_id": "c1", "text": "Alice moved and started painting."}],
                }
            ],
        },
    }


def test_gold_units_preserve_fact_and_freeze_exact_time() -> None:
    payload = {
        "segment_id": "seg-1",
        "gold_facts": [
            {
                "gold_fact_id": "g1",
                "original_text": "Alice moved on May 20, 2023 and started painting.",
                "claim_units": [
                    {
                        "claim_id": "g1:C1",
                        "claim_text": "Alice moved.",
                        "required_time": {
                            "required": True,
                            "normalized_value": "2023-05-20",
                            "resolution": "day",
                            "surface_form": "May 20, 2023",
                        },
                    }
                ],
            }
        ],
    }
    parsed = parse_gold_evaluation_units(json.dumps(payload), _reference_row())
    assert parsed["gold_facts"][0]["claim_units"][0]["required_time"]["required"] is True


def test_set_judge_accepts_missing_exact_time_as_explicit_failure() -> None:
    payload = {
        "segment_id": "seg-1",
        "candidate_set_results": [
            {
                "set_id": "A",
                "candidate_assessments": [
                    {
                        "candidate_id": "c1",
                        "semantic_status": "PARTIALLY_SUPPORTED",
                        "supported_content": ["Alice moved and started painting."],
                        "unsupported_or_incorrect_content": ["The required exact move date is omitted."],
                    }
                ],
                "gold_claim_assessments": [
                    {
                        "gold_fact_id": "g1",
                        "claim_id": "g1:C1",
                        "content_status": "COVERED",
                        "covering_candidate_ids": ["c1"],
                        "time_status": "MISSING_REQUIRED_EXACT_TIME",
                        "covered_content": ["Alice moved."],
                        "missing_or_incorrect_content": ["May 20, 2023 is missing."],
                    },
                    {
                        "gold_fact_id": "g1",
                        "claim_id": "g1:C2",
                        "content_status": "COVERED",
                        "covering_candidate_ids": ["c1"],
                        "time_status": "NOT_APPLICABLE",
                        "covered_content": ["Alice started painting."],
                        "missing_or_incorrect_content": [],
                    },
                ],
            }
        ],
    }
    parsed = parse_segment_set_judgment(json.dumps(payload), _task())
    first = parsed["candidate_set_results"][0]["gold_claim_assessments"][0]
    assert first["content_status"] == "COVERED"
    assert first["time_status"] == "MISSING_REQUIRED_EXACT_TIME"


def test_set_judge_rejects_not_applicable_for_required_time() -> None:
    task = _task()
    payload = {
        "segment_id": "seg-1",
        "candidate_set_results": [{
            "set_id": "A",
            "candidate_assessments": [{
                "candidate_id": "c1", "semantic_status": "SUPPORTED",
                "supported_content": [], "unsupported_or_incorrect_content": [],
            }],
            "gold_claim_assessments": [
                {
                    "gold_fact_id": "g1", "claim_id": "g1:C1",
                    "content_status": "COVERED", "covering_candidate_ids": ["c1"],
                    "time_status": "NOT_APPLICABLE", "covered_content": [],
                    "missing_or_incorrect_content": [],
                },
                {
                    "gold_fact_id": "g1", "claim_id": "g1:C2",
                    "content_status": "COVERED", "covering_candidate_ids": ["c1"],
                    "time_status": "NOT_APPLICABLE", "covered_content": [],
                    "missing_or_incorrect_content": [],
                },
            ],
        }],
    }
    with pytest.raises(ValueError, match="time_status contradicts"):
        parse_segment_set_judgment(json.dumps(payload), task)


def test_pilot_selection_round_robins_conversations() -> None:
    rows = [
        {"sample_id": sample_id, "segment_id": f"{sample_id}-{index}"}
        for sample_id in ("conv-1", "conv-2", "conv-3")
        for index in range(4)
    ]
    selected = _take_stratified(rows, 6)
    assert [row["sample_id"] for row in selected] == [
        "conv-1", "conv-2", "conv-3", "conv-1", "conv-2", "conv-3"
    ]

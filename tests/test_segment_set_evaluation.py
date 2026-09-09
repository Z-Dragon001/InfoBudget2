from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infobudget.quality_router.segment_set_evaluation import (
    _take_stratified,
    _usage_totals,
    _recover_gold_archives,
    parse_gold_evaluation_units,
    parse_segment_set_judgment,
)
from infobudget.rl_router.ledger import SqliteLedger


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
                "gold_fact_id": "gold_fact_id",
                "original_text": "Alice moved on May 20, 2023 and started painting.",
                "claim_units": [
                    {
                        "claim_id": "gold_fact_id:C1",
                        "claim_text": "Alice moved.",
                        "required_time": {
                            "required": "true",
                            "normalized_value": "2023-05-20",
                            "resolution": "day",
                            "surface_form": "May 20, 2023",
                        },
                    },
                    {
                        "claim_id": "gold_fact_id:C1",
                        "claim_text": "Alice started painting.",
                        "required_time": {
                            "required": True,
                            "normalized_value": None,
                            "resolution": None,
                            "surface_form": None,
                        },
                    },
                ],
            }
        ],
    }
    parsed = parse_gold_evaluation_units(json.dumps(payload), _reference_row())
    assert [
        unit["claim_id"]
        for unit in parsed["gold_facts"][0]["claim_units"]
    ] == ["g1:C1", "g1:C2"]
    assert parsed["gold_facts"][0]["claim_units"][0]["required_time"]["required"] is True
    assert parsed["gold_facts"][0]["claim_units"][1]["required_time"] == {
        "required": False,
        "normalized_value": None,
        "resolution": None,
        "surface_form": None,
    }


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


def test_empty_raw_call_directory_has_zero_usage(tmp_path: Path) -> None:
    assert _usage_totals(tmp_path) == {
        "logical_api_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_parser_repair_recovers_archived_gold_response_without_api(tmp_path: Path) -> None:
    response = {
        "segment_id": "seg-1",
        "gold_facts": [{
            "gold_fact_id": "g1",
            "original_text": "Alice moved on May 20, 2023 and started painting.",
            "claim_units": [{
                "claim_id": "g1:C1", "claim_text": "Alice moved.",
                "required_time": {
                    "required": True, "normalized_value": "2023-05-20",
                    "resolution": "date", "surface_form": "May 20, 2023",
                },
            }],
        }],
    }
    raw = tmp_path / "raw_calls"
    raw.mkdir()
    (raw / "failed.json").write_text(json.dumps({
        "status": "invalid_semantic_response", "segment_id": "seg-1",
        "response_content": json.dumps(response),
    }), encoding="utf-8")
    ledger = SqliteLedger(tmp_path / "evaluation.sqlite3", "gold_units", key_fields=("segment_id",))
    recovered = _recover_gold_archives(
        output_dir=tmp_path, references=[_reference_row()], ledger=ledger,
        identity={"prompt_sha256": "prompt-hash"},
        model_spec=SimpleNamespace(effective_model_name="judge-model"),
    )
    assert recovered == 1
    assert ledger.read_all()[0]["recovered_from_raw_call"] == "failed.json"

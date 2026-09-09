from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infobudget.quality_router.segment_set_evaluation import (
    _repair_instruction, _take_stratified, _usage_totals,
    gold_requires_exact_time,
    parse_segment_set_judgment,
)


def _task() -> dict:
    return {
        "segment_id": "seg-1", "dataset_name": "locomo",
        "split": "full", "sample_id": "conv-1",
        "model_by_set": {"A": "model-a"},
        "model_input": {
            "segment_id": "seg-1",
            "segment_text": "Alice moved on May 20, 2023 and paints to relax.",
            "gold_facts": [
                {"gold_fact_id": "g1", "text": "Alice moved on May 20, 2023.", "requires_exact_time": True},
                {"gold_fact_id": "g2", "text": "Alice paints to relax.", "requires_exact_time": False},
            ],
            "candidate_sets": [{
                "set_id": "A",
                "facts": [{"candidate_id": "c1", "text": "Alice moved and paints to relax."}],
            }],
        },
    }


def _response() -> dict:
    return {
        "segment_id": "seg-1",
        "candidate_set_results": [{
            "set_id": "A",
            "candidate_assessments": [{
                "candidate_id": "c1", "semantic_status": "PARTIALLY_SUPPORTED",
                "supported_content": ["Alice moved and paints to relax."],
                "unsupported_or_incorrect_content": ["Exact move date is omitted."],
            }],
            "gold_fact_assessments": [
                {
                    "gold_fact_id": "g1", "coverage_status": "FULL",
                    "covering_candidate_ids": ["c1"],
                    "time_status": "MISSING_REQUIRED_EXACT_TIME",
                    "covered_content": ["Alice moved."],
                    "missing_or_incorrect_content": ["May 20, 2023 is missing."],
                },
                {
                    "gold_fact_id": "g2", "coverage_status": "FULL",
                    "covering_candidate_ids": ["c1"],
                    "time_status": "NOT_APPLICABLE",
                    "covered_content": ["Alice paints to relax."],
                    "missing_or_incorrect_content": [],
                },
            ],
        }],
    }


def test_gold_exact_time_detection_is_conservative() -> None:
    assert gold_requires_exact_time("Alice moved on May 20, 2023.") is True
    assert gold_requires_exact_time("Alice moved in 2023.") is True
    assert gold_requires_exact_time("Alice currently paints.") is False
    assert gold_requires_exact_time("Alice moved last month.") is False


def test_set_judge_accepts_partial_credit_with_time_failure() -> None:
    parsed = parse_segment_set_judgment(json.dumps(_response()), _task())
    first = parsed["candidate_set_results"][0]["gold_fact_assessments"][0]
    assert first["coverage_status"] == "FULL"
    assert first["time_status"] == "MISSING_REQUIRED_EXACT_TIME"


def test_set_judge_rejects_not_applicable_for_exact_time() -> None:
    response = _response()
    response["candidate_set_results"][0]["gold_fact_assessments"][0]["time_status"] = "NOT_APPLICABLE"
    with pytest.raises(ValueError, match="time_status contradicts"):
        parse_segment_set_judgment(json.dumps(response), _task())


def test_set_judge_reports_all_detectable_semantic_errors() -> None:
    task = _task()
    task["model_by_set"]["B"] = "model-b"
    task["model_input"]["candidate_sets"].append({
        "set_id": "B",
        "facts": [{"candidate_id": "c2", "text": "Alice moved."}],
    })
    response = _response()
    second_set = copy.deepcopy(response["candidate_set_results"][0])
    second_set["set_id"] = "B"
    second_set["candidate_assessments"][0]["candidate_id"] = "c2"
    for gold in second_set["gold_fact_assessments"]:
        gold["covering_candidate_ids"] = [
            "c2" if candidate_id == "c1" else candidate_id
            for candidate_id in gold["covering_candidate_ids"]
        ]
    response["candidate_set_results"].append(second_set)
    gold = response["candidate_set_results"][0]["gold_fact_assessments"]
    gold[0]["time_status"] = "NOT_APPLICABLE"
    second_set["gold_fact_assessments"][1]["coverage_status"] = "SUPPORTED"
    with pytest.raises(ValueError) as exc_info:
        parse_segment_set_judgment(json.dumps(response), task)
    message = str(exc_info.value)
    assert "semantic validation failed with 2 error(s)" in message
    assert "g1: time_status contradicts Gold time policy" in message
    assert "g2: invalid coverage_status 'SUPPORTED'" in message


def test_repair_instruction_includes_previous_json_and_full_policy() -> None:
    previous = '{"segment_id":"seg-1","candidate_set_results":[]}'
    prompt = _repair_instruction(_task(), "example validation error", previous)
    assert previous in prompt
    assert "example validation error" in prompt
    assert '"requires_exact_time": true' in prompt
    assert (
        '"allowed_time_statuses": ["PASS", "MISSING_REQUIRED_EXACT_TIME", '
        '"CONTRADICTED_TIME"]' in prompt
    )
    assert "Never use SUPPORTED or PARTIALLY_SUPPORTED as coverage_status" in prompt
    assert "2023-05-07 and May 7, 2023" in prompt


def test_set_judge_rejects_full_coverage_without_candidate_ids() -> None:
    response = _response()
    response["candidate_set_results"][0]["gold_fact_assessments"][1]["covering_candidate_ids"] = []
    with pytest.raises(ValueError, match="covered Gold needs Candidate IDs"):
        parse_segment_set_judgment(json.dumps(response), _task())


def test_set_judge_accepts_valid_gold_id_reordering_without_remapping() -> None:
    response = _response()
    gold = response["candidate_set_results"][0]["gold_fact_assessments"]
    gold.reverse()
    parsed = parse_segment_set_judgment(json.dumps(response), _task())
    returned = parsed["candidate_set_results"][0]["gold_fact_assessments"]
    assert [item["gold_fact_id"] for item in returned] == ["g2", "g1"]
    assert returned[0]["time_status"] == "NOT_APPLICABLE"


def test_set_judge_rejects_duplicate_gold_ids_instead_of_guessing_mapping() -> None:
    response = _response()
    gold = response["candidate_set_results"][0]["gold_fact_assessments"]
    gold[1]["gold_fact_id"] = "g1"
    with pytest.raises(
        ValueError, match=r"missing=\['g2'\].*duplicated=\['g1'\]"
    ):
        parse_segment_set_judgment(json.dumps(response), _task())


def test_pilot_selection_round_robins_conversations() -> None:
    rows = [
        {"sample_id": sample_id, "segment_id": f"{sample_id}-{index}"}
        for sample_id in ("conv-1", "conv-2", "conv-3") for index in range(4)
    ]
    selected = _take_stratified(rows, 6)
    assert [row["sample_id"] for row in selected] == [
        "conv-1", "conv-2", "conv-3", "conv-1", "conv-2", "conv-3"
    ]


def test_empty_raw_call_directory_has_zero_usage(tmp_path: Path) -> None:
    assert _usage_totals(tmp_path) == {
        "logical_api_call_count": 0, "input_tokens": 0, "output_tokens": 0,
    }

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infobudget.quality_router.equivalence_judge import (
    parse_equivalence_decisions,
    plan_equivalence_judging,
    run_equivalence_judging,
)
from infobudget.rl_router.api import LLMResponse
from infobudget.schemas import ModelSpec, PriceSpec


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _pair(pair_id: str = "p1") -> dict:
    return {
        "pair_id": pair_id,
        "dataset_name": "locomo",
        "split": "full",
        "sample_id": "conv-1",
        "segment_id": "seg-1",
        "model_id": "model-a",
        "candidate_fact_id": "c1",
        "candidate_fact_text": "Alice moved.",
        "candidate_source_turn_ids": [1],
        "reference_fact_id": "r1",
        "reference_fact_text": "Alice relocated.",
        "reference_source_turn_ids": [1],
    }


def _model() -> ModelSpec:
    return ModelSpec(
        deploy="api",
        backend="openai_compatible",
        model_name="judge",
        tokenizer_name="judge",
        max_context_tokens=4096,
        max_output_tokens=1024,
        tensor_parallel_size=1,
        dtype="n/a",
    )


def test_plan_is_read_only_and_groups_pairs(tmp_path: Path) -> None:
    segments = tmp_path / "segments"
    pairs = tmp_path / "pairs.jsonl"
    prompt = tmp_path / "prompt.txt"
    _write_jsonl(
        segments / "segments.jsonl",
        [
            {
                "dataset_name": "locomo",
                "split": "full",
                "sample_id": "conv-1",
                "segment_id": "seg-1",
                "turn_ids": [1],
                "text": "[2023-01-01, Sun] 0.Alice: I moved.",
            }
        ],
    )
    _write_jsonl(pairs, [_pair("p1"), {**_pair("p2"), "candidate_fact_id": "c2"}])
    prompt.write_text("Judge strictly.", encoding="utf-8")
    result = plan_equivalence_judging(
        segments_path=segments,
        pairs_path=pairs,
        pairs_manifest_path=None,
        prompt_path=prompt,
        model_spec=_model(),
        price=PriceSpec(0.1, 0.2),
        batch_size=1,
    )
    assert result["paid_api_called"] is False
    assert result["pair_count"] == 2
    assert result["batch_count"] == 2


def test_parse_decisions_supports_asymmetric_candidate_coverage() -> None:
    batch = [_pair()]
    content = json.dumps(
        {
            "decisions": [
                {
                    "pair_id": "p1",
                    "candidate_fully_grounded": True,
                    "reference_fully_grounded": True,
                    "candidate_entails_reference": True,
                    "reference_entails_candidate": False,
                    "material_overlap": True,
                }
            ]
        }
    )
    decision = parse_equivalence_decisions(content, batch)[0]
    assert decision["strict_equivalent"] is False
    assert decision["candidate_covers_reference"] is True
    broken = json.loads(content)
    broken["decisions"][0]["material_overlap"] = "true"
    with pytest.raises(ValueError, match="material_overlap must be a JSON boolean"):
        parse_equivalence_decisions(json.dumps(broken), batch)


@pytest.mark.parametrize(
    ("fields", "expected_relation"),
    [
        (
            {
                "candidate_fully_grounded": False,
                "reference_fully_grounded": True,
                "candidate_entails_reference": False,
                "reference_entails_candidate": False,
                "material_overlap": False,
            },
            "UNSUPPORTED",
        ),
        (
            {
                "candidate_fully_grounded": True,
                "reference_fully_grounded": True,
                "candidate_entails_reference": True,
                "reference_entails_candidate": True,
                "material_overlap": True,
            },
            "EQUIVALENT",
        ),
        (
            {
                "candidate_fully_grounded": True,
                "reference_fully_grounded": True,
                "candidate_entails_reference": True,
                "reference_entails_candidate": False,
                "material_overlap": True,
            },
            "CANDIDATE_CONTAINS_REFERENCE",
        ),
        (
            {
                "candidate_fully_grounded": True,
                "reference_fully_grounded": True,
                "candidate_entails_reference": False,
                "reference_entails_candidate": True,
                "material_overlap": True,
            },
            "REFERENCE_CONTAINS_CANDIDATE",
        ),
        (
            {
                "candidate_fully_grounded": True,
                "reference_fully_grounded": True,
                "candidate_entails_reference": False,
                "reference_entails_candidate": False,
                "material_overlap": True,
            },
            "PARTIAL_OVERLAP",
        ),
        (
            {
                "candidate_fully_grounded": True,
                "reference_fully_grounded": True,
                "candidate_entails_reference": False,
                "reference_entails_candidate": False,
                "material_overlap": False,
            },
            "DIFFERENT",
        ),
    ],
)
def test_parse_decisions_derives_all_relations(
    fields: dict[str, bool], expected_relation: str
) -> None:
    content = json.dumps({"decisions": [{"pair_id": "p1", **fields}]})
    decision = parse_equivalence_decisions(content, [_pair()])[0]
    assert decision["relation"] == expected_relation


def test_run_exports_complete_judgments_and_resumes(tmp_path: Path) -> None:
    segments = tmp_path / "segments"
    pairs = tmp_path / "pairs.jsonl"
    prompt = tmp_path / "prompt.txt"
    output_dir = tmp_path / "judge"
    output = tmp_path / "judgments.jsonl"
    _write_jsonl(
        segments / "segments.jsonl",
        [
            {
                "dataset_name": "locomo",
                "split": "full",
                "sample_id": "conv-1",
                "segment_id": "seg-1",
                "turn_ids": [1],
                "text": "[2023-01-01, Sun] 0.Alice: I moved.",
            }
        ],
    )
    _write_jsonl(pairs, [_pair()])
    prompt.write_text("Judge strictly.", encoding="utf-8")

    class FakeClient:
        calls = 0

        def complete(self, **kwargs) -> LLMResponse:
            self.calls += 1
            assert (
                "<SOURCE_TURN_ID=1> [2023-01-01, Sun] Alice: I moved."
                in kwargs["prompt"]
            )
            assert "0.Alice" not in kwargs["prompt"]
            return LLMResponse(
                content=json.dumps(
                    {
                        "decisions": [
                            {
                                "pair_id": "p1",
                                "candidate_fully_grounded": True,
                                "reference_fully_grounded": True,
                                "candidate_entails_reference": True,
                                "reference_entails_candidate": True,
                                "material_overlap": True,
                            }
                        ]
                    }
                ),
                input_tokens=100,
                output_tokens=20,
                latency_ms=1,
            )

    client = FakeClient()
    kwargs = dict(
        segments_path=segments,
        pairs_path=pairs,
        pairs_manifest_path=None,
        prompt_path=prompt,
        output_dir=output_dir,
        output_path=output,
        model_spec=_model(),
        price=PriceSpec(0.1, 0.2),
        client=client,
        batch_size=32,
    )
    first = run_equivalence_judging(**kwargs)
    second = run_equivalence_judging(**kwargs)
    assert first["run_complete"] is True
    assert second["run_complete"] is True
    assert client.calls == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_run_recovers_valid_archived_decisions_and_retries_only_missing_id(
    tmp_path: Path,
) -> None:
    segments = tmp_path / "segments"
    pairs = tmp_path / "pairs.jsonl"
    prompt = tmp_path / "prompt.txt"
    output_dir = tmp_path / "judge"
    output = tmp_path / "judgments.jsonl"
    first = _pair("p1")
    second = {**_pair("p2"), "candidate_fact_id": "c2"}
    _write_jsonl(
        segments / "segments.jsonl",
        [
            {
                "dataset_name": "locomo",
                "split": "full",
                "sample_id": "conv-1",
                "segment_id": "seg-1",
                "turn_ids": [1],
                "text": "[2023-01-01, Sun] 0.Alice: I moved.",
            }
        ],
    )
    _write_jsonl(pairs, [first, second])
    prompt.write_text("Judge strictly.", encoding="utf-8")
    archived_decision = {
        "pair_id": "p1",
        "candidate_fully_grounded": True,
        "reference_fully_grounded": True,
        "candidate_entails_reference": True,
        "reference_entails_candidate": True,
        "material_overlap": True,
    }
    raw_dir = output_dir / "raw_calls"
    raw_dir.mkdir(parents=True)
    (raw_dir / "old_partial.json").write_text(
        json.dumps(
            {
                "status": "invalid_semantic_response",
                "batch_id": "old-batch",
                "pair_ids": ["p1", "p2"],
                "response_content": json.dumps(
                    {"decisions": [archived_decision]}
                ),
                "archived_at": "2026-09-08T00:00:00+00:00",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        ),
        encoding="utf-8",
    )

    class MissingOnlyClient:
        calls = 0

        def complete(self, **kwargs) -> LLMResponse:
            self.calls += 1
            assert '"pair_id": "p2"' in kwargs["prompt"]
            assert '"pair_id": "p1"' not in kwargs["prompt"]
            return LLMResponse(
                content=json.dumps(
                    {
                        "decisions": [
                            {
                                **archived_decision,
                                "pair_id": "p2",
                            }
                        ]
                    }
                ),
                input_tokens=30,
                output_tokens=10,
                latency_ms=1,
            )

    client = MissingOnlyClient()
    result = run_equivalence_judging(
        segments_path=segments,
        pairs_path=pairs,
        pairs_manifest_path=None,
        prompt_path=prompt,
        output_dir=output_dir,
        output_path=output,
        model_spec=_model(),
        price=PriceSpec(0.1, 0.2),
        client=client,
        batch_size=2,
    )
    assert result["run_complete"] is True
    assert result["completed_decision_count"] == 2
    assert client.calls == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["pair_id"] for row in rows] == ["p1", "p2"]


def test_run_commits_partial_response_and_retries_only_omitted_id(
    tmp_path: Path,
) -> None:
    segments = tmp_path / "segments"
    pairs = tmp_path / "pairs.jsonl"
    prompt = tmp_path / "prompt.txt"
    output_dir = tmp_path / "judge"
    output = tmp_path / "judgments.jsonl"
    first = _pair("p1")
    second = {**_pair("p2"), "candidate_fact_id": "c2"}
    _write_jsonl(
        segments / "segments.jsonl",
        [
            {
                "dataset_name": "locomo",
                "split": "full",
                "sample_id": "conv-1",
                "segment_id": "seg-1",
                "turn_ids": [1],
                "text": "[2023-01-01, Sun] 0.Alice: I moved.",
            }
        ],
    )
    _write_jsonl(pairs, [first, second])
    prompt.write_text("Judge strictly.", encoding="utf-8")

    class PartialThenCompleteClient:
        calls = 0

        def complete(self, **kwargs) -> LLMResponse:
            self.calls += 1
            pair_id = "p1" if self.calls == 1 else "p2"
            if self.calls == 2:
                assert '"pair_id": "p2"' in kwargs["prompt"]
                assert '"pair_id": "p1"' not in kwargs["prompt"]
            return LLMResponse(
                content=json.dumps(
                    {
                        "decisions": [
                            {
                                "pair_id": pair_id,
                                "candidate_fully_grounded": True,
                                "reference_fully_grounded": True,
                                "candidate_entails_reference": True,
                                "reference_entails_candidate": True,
                                "material_overlap": True,
                            }
                        ]
                    }
                ),
                input_tokens=30,
                output_tokens=10,
                latency_ms=1,
            )

    client = PartialThenCompleteClient()
    result = run_equivalence_judging(
        segments_path=segments,
        pairs_path=pairs,
        pairs_manifest_path=None,
        prompt_path=prompt,
        output_dir=output_dir,
        output_path=output,
        model_spec=_model(),
        price=PriceSpec(0.1, 0.2),
        client=client,
        batch_size=2,
    )
    assert result["run_complete"] is True
    assert result["completed_decision_count"] == 2
    assert result["partial_response_count"] == 1
    assert client.calls == 2


def test_run_retries_only_pair_with_invalid_primitive_field(
    tmp_path: Path,
) -> None:
    segments = tmp_path / "segments"
    pairs = tmp_path / "pairs.jsonl"
    prompt = tmp_path / "prompt.txt"
    output_dir = tmp_path / "judge"
    output = tmp_path / "judgments.jsonl"
    first = _pair("p1")
    second = {**_pair("p2"), "candidate_fact_id": "c2"}
    _write_jsonl(
        segments / "segments.jsonl",
        [
            {
                "dataset_name": "locomo",
                "split": "full",
                "sample_id": "conv-1",
                "segment_id": "seg-1",
                "turn_ids": [1],
                "text": "[2023-01-01, Sun] 0.Alice: I moved.",
            }
        ],
    )
    _write_jsonl(pairs, [first, second])
    prompt.write_text("Judge strictly.", encoding="utf-8")
    valid = {
        "candidate_fully_grounded": True,
        "reference_fully_grounded": True,
        "candidate_entails_reference": True,
        "reference_entails_candidate": True,
        "material_overlap": True,
    }

    class InconsistentThenCompleteClient:
        calls = 0

        def complete(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                decisions = [
                    {"pair_id": "p1", **valid},
                    {"pair_id": "p2", **valid, "material_overlap": "true"},
                ]
            else:
                assert '"pair_id": "p2"' in kwargs["prompt"]
                assert '"pair_id": "p1"' not in kwargs["prompt"]
                decisions = [{"pair_id": "p2", **valid}]
            return LLMResponse(
                content=json.dumps({"decisions": decisions}),
                input_tokens=30,
                output_tokens=10,
                latency_ms=1,
            )

    client = InconsistentThenCompleteClient()
    result = run_equivalence_judging(
        segments_path=segments,
        pairs_path=pairs,
        pairs_manifest_path=None,
        prompt_path=prompt,
        output_dir=output_dir,
        output_path=output,
        model_spec=_model(),
        price=PriceSpec(0.1, 0.2),
        client=client,
        batch_size=2,
    )
    assert result["run_complete"] is True
    assert result["completed_decision_count"] == 2
    assert result["partial_response_count"] == 1
    assert client.calls == 2

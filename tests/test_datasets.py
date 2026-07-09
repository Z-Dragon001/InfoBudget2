"""功能：测试 LoCoMo / LongMemEval 预处理、存储与评估。
输入：合成原始数据文件。
输出：processed 工件与评估结果断言。
依赖：pytest、项目 datasets/evaluation 模块。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from infobudget.config import ProjectBundle, load_project_bundle
from infobudget.datasets.longmemeval import LongMemEvalPreprocessor
from infobudget.datasets.loader import DatasetLoader
from infobudget.datasets.preprocess import DatasetPreprocessManager
from infobudget.evaluation.dataset_runner import DatasetEvaluationRunner


def _offline_bundle(bundle: ProjectBundle, root_dir: Path) -> ProjectBundle:
    config = replace(
        bundle.config,
        extractor=replace(bundle.config.extractor, mode="mock_joint"),
        evaluation=replace(
            bundle.config.evaluation,
            judge_mode="rule_judge",
            judge_model=None,
            qa_mode="retrieved_top1",
        ),
    )
    return type(bundle)(
        root_dir=root_dir,
        config=config,
        weights=bundle.weights,
        models=bundle.models,
        prices=bundle.prices,
        prompt_dir=bundle.prompt_dir,
    )


def test_preprocess_locomo_and_longmemeval(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    raw_locomo_dir = tmp_path / "datasets" / "raw" / "locomo"
    raw_longmem_dir = tmp_path / "datasets" / "raw" / "longmemeval"
    raw_locomo_dir.mkdir(parents=True, exist_ok=True)
    raw_longmem_dir.mkdir(parents=True, exist_ok=True)

    locomo_payload = [
        {
            "sample_id": "locomo_case_1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {
                        "speaker": "Alice",
                        "dia_id": "D1:1",
                        "text": "I am building the InfoBudget dataset pipeline.",
                        "img_url": ["https://example.com/a.jpg"],
                        "blip_caption": "a whiteboard with diagrams",
                        "query": "whiteboard project plan",
                    },
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "Then we need preprocessing and evaluation modules."},
                ],
            },
            "qa": [
                {
                    "question": "What is Alice building?",
                    "answer": "InfoBudget dataset pipeline",
                    "evidence": ["D1:1"],
                    "category": 1,
                }
            ],
            "event_summary": {},
            "observation": {},
            "session_summary": {},
        }
    ]
    longmem_payload = [
        {
            "question_id": "longmem_case_1",
            "question_type": "single-session-user",
            "question": "What do I need?",
            "question_date": "2023/05/30 (Tue) 23:40",
            "answer": "a unified loader",
            "answer_session_ids": ["sess_1"],
            "haystack_session_ids": ["sess_1"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I need a unified loader."},
                    {"role": "assistant", "content": "We should also generate processed jsonl files."},
                ]
            ],
        }
    ]

    with (raw_locomo_dir / "locomo10.json").open("w", encoding="utf-8") as handle:
        json.dump(locomo_payload, handle, ensure_ascii=False)
    with (raw_longmem_dir / "longmemeval_s_cleaned.json").open("w", encoding="utf-8") as handle:
        json.dump(longmem_payload, handle, ensure_ascii=False)

    manager = DatasetPreprocessManager(bundle.config.dataset, tmp_path)
    locomo_summary = manager.preprocess_dataset("locomo")
    longmem_summary = manager.preprocess_dataset("longmemeval")

    assert locomo_summary["full"] == 1
    assert longmem_summary["full"] == 1

    loader = DatasetLoader(bundle.config.dataset, tmp_path)
    locomo_manifest = loader.load_manifest("locomo", "full")
    longmem_manifest = loader.load_manifest("longmemeval", "full")
    loaded_locomo = loader.load("locomo", "full")
    loaded_longmem = loader.load("longmemeval", "full")

    assert locomo_manifest["num_questions"] == 1
    assert longmem_manifest["num_sessions"] == 1
    assert loaded_locomo[0].sessions[0].timestamp == "2023-05-08 13:56:00"
    assert loaded_locomo[0].sessions[0].turns[0].timestamp == "2023-05-08T13:56:00.000"
    assert loaded_locomo[0].sessions[0].turns[1].timestamp == "2023-05-08T13:56:01.000"
    assert loaded_locomo[0].sessions[0].turns[0].metadata["weekday"] == "Mon"
    assert loaded_locomo[0].sessions[0].turns[0].metadata["timestamp_source"] == "synthetic_turn_from_session"
    assert loaded_locomo[0].sessions[0].turns[0].metadata["session_id"] == "session_1"
    assert loaded_locomo[0].qa_pairs[0].evidence_session_ids == ["session_1"]
    assert loaded_locomo[0].sessions[0].turns[0].metadata["dia_id"] == "D1:1"
    assert loaded_locomo[0].sessions[0].turns[0].metadata["blip_caption"] == "a whiteboard with diagrams"
    assert loaded_locomo[0].sessions[0].turns[0].metadata["query"] == "whiteboard project plan"
    assert loaded_longmem[0].qa_pairs[0].question_type == "single-session-user"
    assert loaded_longmem[0].sessions[0].timestamp == "2023-05-20 02:21:00"
    assert loaded_longmem[0].sessions[0].turns[0].timestamp == "2023-05-20T02:21:00.000"
    assert loaded_longmem[0].sessions[0].turns[1].timestamp == "2023-05-20T02:21:00.500"
    assert loaded_longmem[0].sessions[0].turns[0].metadata["weekday"] == "Sat"
    assert loaded_longmem[0].sessions[0].turns[0].metadata["timestamp_source"] == "synthetic_turn_from_session"


def test_locomo_preprocess_ignores_timestamp_only_session_keys(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    raw_locomo_dir = tmp_path / "datasets" / "raw" / "locomo"
    raw_locomo_dir.mkdir(parents=True, exist_ok=True)

    locomo_payload = [
        {
            "sample_id": "locomo_case_sparse_sessions",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "Only one real session exists."},
                ],
                "session_20_date_time": "4:10 pm on 26 October, 2023",
            },
            "qa": [],
            "event_summary": {},
            "observation": {},
            "session_summary": {},
        }
    ]

    with (raw_locomo_dir / "locomo10.json").open("w", encoding="utf-8") as handle:
        json.dump(locomo_payload, handle, ensure_ascii=False)

    manager = DatasetPreprocessManager(bundle.config.dataset, tmp_path)
    manager.preprocess_dataset("locomo")

    loader = DatasetLoader(bundle.config.dataset, tmp_path)
    loaded_locomo = loader.load("locomo", "full")

    assert len(loaded_locomo[0].sessions) == 1
    assert loaded_locomo[0].sessions[0].session_id == "session_1"
    assert loaded_locomo[0].sessions[0].turns[0].text == "Only one real session exists."
    assert loaded_locomo[0].sessions[0].turns[0].timestamp == "2023-05-08T13:56:00.000"


def test_longmemeval_abs_question_id_matches_lightmem_abstention_rule() -> None:
    assert (
        LongMemEvalPreprocessor._judge_profile_for_type("single-session-user", "longmem_abs_case_1")
        == "longmemeval_abstention"
    )


def test_dataset_evaluation_runner(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    raw_locomo_dir = tmp_path / "datasets" / "raw" / "locomo"
    raw_locomo_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "sample_id": "locomo_eval_1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "We need LoCoMo and LongMemEval preprocessing."},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "We also need a unified dataset loader and evaluation runner."},
                ],
            },
            "qa": [
                {
                    "question": "What do we need?",
                    "answer": "LoCoMo and LongMemEval preprocessing",
                    "evidence": ["D1:1"],
                    "category": 1,
                }
            ],
            "event_summary": {},
            "observation": {},
            "session_summary": {},
        }
    ]
    with (raw_locomo_dir / "locomo10.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)

    manager = DatasetPreprocessManager(bundle.config.dataset, tmp_path)
    manager.preprocess_dataset("locomo")

    custom_bundle = _offline_bundle(bundle, tmp_path)
    runner = DatasetEvaluationRunner(custom_bundle)
    result = runner.evaluate("locomo", "full")

    assert result.dataset_name == "locomo"
    assert result.split == "full"
    assert len(result.predictions) == 1
    assert len(result.retrieval_traces) == 1
    assert (tmp_path / "outputs" / "evaluation" / "locomo" / "full" / "full" / "flat" / "metrics.json").exists()
    assert (tmp_path / "outputs" / "evaluation" / "locomo" / "full" / "full" / "flat" / "retrieval_traces.jsonl").exists()
    full_memory_root = tmp_path / "outputs" / "memory" / "locomo" / "full" / "full" / "flat"
    full_memory_dir = full_memory_root / "locomo_eval_1"
    assert (full_memory_dir / "memory_jsonl" / "memory_entries.jsonl").exists()
    assert (full_memory_dir / "memory_jsonl" / "segments.jsonl").exists()
    assert (full_memory_dir / "memory_jsonl" / "cost_logs.jsonl").exists()
    assert (full_memory_dir / "qdrant").exists()
    assert (full_memory_root / "build_manifest.json").exists()

    entropy_runner = DatasetEvaluationRunner(custom_bundle, scoring_mode="entropy_only")
    entropy_runner.evaluate("locomo", "full")

    assert (tmp_path / "outputs" / "evaluation" / "locomo" / "full" / "entropy_only" / "flat" / "metrics.json").exists()
    entropy_memory_root = tmp_path / "outputs" / "memory" / "locomo" / "full" / "entropy_only" / "flat"
    entropy_memory_dir = entropy_memory_root / "locomo_eval_1"
    assert (entropy_memory_dir / "memory_jsonl" / "memory_entries.jsonl").exists()
    assert (entropy_memory_dir / "memory_jsonl" / "segments.jsonl").exists()
    assert (entropy_memory_dir / "memory_jsonl" / "cost_logs.jsonl").exists()
    assert (entropy_memory_dir / "qdrant").exists()
    with (tmp_path / "outputs" / "evaluation" / "locomo" / "full" / "entropy_only" / "flat" / "run_manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        entropy_manifest = json.load(handle)
    assert entropy_manifest["split"] == "full"
    assert entropy_manifest["output_label"] == "full/entropy_only/flat"
    assert entropy_manifest["scoring_mode"] == "entropy_only"
    assert entropy_manifest["extraction_mode"] == "flat"
    assert entropy_manifest["memory_output_dir"] == str(entropy_memory_root)
    assert entropy_manifest["memory_eval_mode"] == "evaluate_existing_memories"

    limited_runner = DatasetEvaluationRunner(custom_bundle, scoring_mode="concept_density_only")
    limited_runner.evaluate("locomo", "full", limit=1)
    with (tmp_path / "outputs" / "evaluation" / "locomo" / "full" / "concept_density_only" / "flat" / "run_manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        limited_manifest = json.load(handle)
    assert limited_manifest["num_examples"] == 1
    assert limited_manifest["sample_limit"] == 1

    separated_runner = DatasetEvaluationRunner(custom_bundle, scoring_mode="actionability_only")
    build_result = separated_runner.build_memories("locomo", "full", limit=1)
    eval_result = separated_runner.evaluate_existing_memories("locomo", "full", limit=1)
    separated_memory_root = tmp_path / "outputs" / "memory" / "locomo" / "full" / "actionability_only" / "flat"
    assert build_result.memory_root == separated_memory_root
    assert eval_result.dataset_name == "locomo"
    assert (separated_memory_root / "build_manifest.json").exists()
    assert (tmp_path / "outputs" / "evaluation" / "locomo" / "full" / "actionability_only" / "flat" / "metrics.json").exists()

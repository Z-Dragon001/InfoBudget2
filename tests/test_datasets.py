"""Processed-v3 dataset preprocessing tests."""

from __future__ import annotations

import json
from pathlib import Path

from infobudget.config import load_project_bundle
from infobudget.datasets.loader import DatasetLoader
from infobudget.datasets.preprocess import DatasetPreprocessManager


def test_preprocess_locomo_and_longmemeval_to_v3_artifacts(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    locomo_dir = tmp_path / "datasets" / "raw" / "locomo"
    longmem_dir = tmp_path / "datasets" / "raw" / "longmemeval"
    locomo_dir.mkdir(parents=True)
    longmem_dir.mkdir(parents=True)
    (locomo_dir / "locomo10.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "locomo-1",
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1_date_time": "1:56 pm on 8 May, 2023",
                        "session_1": [
                            {
                                "speaker": "Alice",
                                "dia_id": "D1:1",
                                "text": "I visited Paris.",
                                "blip_caption": "the Eiffel Tower",
                            },
                            {"speaker": "Bob", "dia_id": "D1:2", "text": "That sounds wonderful."},
                        ],
                    },
                    "qa": [{"question": "Where did Alice visit?", "answer": "Paris", "evidence": ["D1:1"], "category": 1}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (longmem_dir / "longmemeval_s_cleaned.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "longmem-1",
                    "question_type": "single-session-user",
                    "question": "What do I need?",
                    "answer": "a router",
                    "answer_session_ids": ["s1"],
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                    "haystack_sessions": [[{"role": "user", "content": "I need a router."}]],
                }
            ]
        ),
        encoding="utf-8",
    )
    manager = DatasetPreprocessManager(bundle.config.dataset, tmp_path)
    assert manager.preprocess_dataset("locomo") == {"full": 1}
    assert manager.preprocess_dataset("longmemeval") == {"full": 1}

    loader = DatasetLoader(bundle.config.dataset, tmp_path)
    locomo = loader.load("locomo", "full")[0]
    manifest = loader.load_manifest("locomo", "full")
    assert manifest["schema_version"] == "processed_v3"
    assert manifest["num_turns"] == 2
    assert Path(manifest["files"]["turns"]).is_file()
    assert manifest["source_file_hashes"]["locomo10.json"]
    assert loader.load("locomo", "full", {"missing"}) == []
    first_turn = locomo.dialogue[0]
    assert first_turn.timestamp == "2023-05-08T13:56:00.000"
    assert first_turn.metadata["weekday"] == "Mon"
    assert first_turn.text.count("(image description:") == 1
    assert first_turn.memory_text().count("(image description:") == 1
    assert locomo.qa_pairs[0].evidence_turn_ids == [1]


def test_sparse_locomo_session_keys_do_not_create_empty_sessions(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    raw_dir = tmp_path / "datasets" / "raw" / "locomo"
    raw_dir.mkdir(parents=True)
    (raw_dir / "locomo10.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": "sparse",
                    "conversation": {
                        "session_1_date_time": "1:56 pm on 8 May, 2023",
                        "session_1": [{"speaker": "Alice", "text": "Only a real session."}],
                        "session_20_date_time": "4:10 pm on 26 October, 2023",
                    },
                    "qa": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    DatasetPreprocessManager(bundle.config.dataset, tmp_path).preprocess_dataset("locomo")
    example = DatasetLoader(bundle.config.dataset, tmp_path).load("locomo", "full")[0]
    assert [session.session_id for session in example.sessions] == ["session_1"]

"""功能：实现 LongMemEval 数据集预处理。
输入：LongMemEval 原始记录。
输出：统一 DatasetDialogueExample，并保留 question-centric haystack 结构。
依赖：datasets.base、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

from typing import Any

from infobudget.datasets.base import (
    BaseDatasetPreprocessor,
    build_sessions_from_flat_turns,
    normalize_qa_pairs,
    parse_longmemeval_timestamp,
)
from infobudget.schemas import DatasetDialogueExample


class LongMemEvalPreprocessor(BaseDatasetPreprocessor):
    """LongMemEval 预处理器。"""

    dataset_name = "longmemeval"

    def normalize_record(
        self,
        raw_record: dict[str, Any],
        *,
        split: str,
        sample_index: int,
    ) -> DatasetDialogueExample:
        question_id = str(raw_record.get("question_id") or raw_record.get("sample_id") or f"longmemeval_{split}_{sample_index:06d}")
        session_ids = [str(item) for item in raw_record.get("haystack_session_ids", [])]
        raw_timestamps = [str(item) for item in raw_record.get("haystack_dates", [])]
        raw_sessions = raw_record.get("haystack_sessions", [])
        sessions, dialogue = build_sessions_from_flat_turns(
            raw_sessions,
            session_ids=session_ids,
            raw_timestamps=raw_timestamps,
            parser=parse_longmemeval_timestamp,
            synthesize_turn_timestamps=True,
            turn_timestamp_step_ms=500,
        )

        raw_qa = {
            "question": raw_record.get("question", ""),
            "answer": raw_record.get("answer", ""),
            "question_type": raw_record.get("question_type", ""),
            "question_date": raw_record.get("question_date"),
            "answer_session_ids": raw_record.get("answer_session_ids", []),
            "judge_profile": self._judge_profile_for_type(
                str(raw_record.get("question_type", "")),
                question_id,
            ),
            "is_unanswerable": "abs" in question_id,
        }
        qa_pairs = normalize_qa_pairs(
            [raw_qa],
            question_id,
            default_judge_profile=raw_qa["judge_profile"],
            category_formatter=lambda value: str(value),
        )
        if qa_pairs:
            qa_pairs[0].question_id = question_id

        metadata = {
            "question_id": question_id,
            "question_type": raw_record.get("question_type", ""),
            "question_date": raw_record.get("question_date"),
            "answer_session_ids": [str(item) for item in raw_record.get("answer_session_ids", [])],
            "haystack_session_ids": session_ids,
            "haystack_dates": raw_timestamps,
            "session_count": len(sessions),
        }
        return DatasetDialogueExample(
            sample_id=question_id,
            dataset_name=self.dataset_name,
            split=split,
            sessions=sessions,
            dialogue=dialogue,
            qa_pairs=qa_pairs,
            metadata=metadata,
        )

    @staticmethod
    def _judge_profile_for_type(question_type: str, question_id: str) -> str:
        if "abs" in question_id:
            return "longmemeval_abstention"
        normalized = question_type.strip().lower()
        mapping = {
            "single-session-user": "longmemeval_single_session",
            "single-session-assistant": "longmemeval_single_session",
            "multi-session": "longmemeval_single_session",
            "temporal-reasoning": "longmemeval_temporal_reasoning",
            "knowledge-update": "longmemeval_knowledge_update",
            "single-session-preference": "longmemeval_preference",
        }
        return mapping.get(normalized, "longmemeval_generic")

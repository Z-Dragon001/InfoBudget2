"""功能：实现 LoCoMo 数据集预处理。
输入：LoCoMo 原始记录。
输出：统一 DatasetDialogueExample，并保留 session / evidence 结构。
依赖：datasets.base、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

import re
from typing import Any

from infobudget.datasets.base import (
    BaseDatasetPreprocessor,
    build_sessions_from_flat_turns,
    normalize_qa_pairs,
    parse_locomo_timestamp,
)
from infobudget.schemas import DatasetDialogueExample


class LOCOMOPreprocessor(BaseDatasetPreprocessor):
    """LoCoMo 预处理器。"""

    dataset_name = "locomo"
    _SESSION_KEY_PATTERN = re.compile(r"^session_(\d+)$")

    def normalize_record(
        self,
        raw_record: dict[str, Any],
        *,
        split: str,
        sample_index: int,
    ) -> DatasetDialogueExample:
        sample_id = str(raw_record.get("sample_id") or raw_record.get("id") or f"locomo_{split}_{sample_index:06d}")
        conversation = raw_record.get("conversation") or {}
        session_numbers = sorted(
            {
                int(match.group(1))
                for key in conversation.keys()
                for match in [self._SESSION_KEY_PATTERN.fullmatch(key)]
                if match is not None
            }
        )
        raw_sessions = [conversation.get(f"session_{number}", []) for number in session_numbers]
        session_ids = [f"session_{number}" for number in session_numbers]
        raw_timestamps = [conversation.get(f"session_{number}_date_time") for number in session_numbers]
        sessions, dialogue = build_sessions_from_flat_turns(
            raw_sessions,
            session_ids=session_ids,
            raw_timestamps=raw_timestamps,
            parser=parse_locomo_timestamp,
            synthesize_turn_timestamps=True,
            turn_timestamp_step_ms=1000,
        )
        qa_pairs = normalize_qa_pairs(
            raw_record.get("qa", []),
            sample_id,
            default_judge_profile="locomo_qa",
            category_formatter=lambda value: f"category_{value}" if value != "" else "",
        )
        qa_by_text = {
            (pair.question, pair.answer): pair
            for pair in qa_pairs
        }
        dia_to_turn_id = self._build_dia_to_turn_map(raw_sessions, sessions)
        for raw_qa in raw_record.get("qa", []):
            key = (str(raw_qa.get("question", "")), str(raw_qa.get("answer", "")))
            pair = qa_by_text.get(key)
            if pair is None:
                continue
            pair.evidence_turn_refs = [str(item) for item in raw_qa.get("evidence", [])]
            pair.evidence_session_ids = sorted({self._session_from_dia_ref(item) for item in pair.evidence_turn_refs if self._session_from_dia_ref(item)})
            pair.evidence_turn_ids = [dia_to_turn_id[item] for item in pair.evidence_turn_refs if item in dia_to_turn_id]
            pair.is_unanswerable = str(raw_qa.get("category", "")) == "5"

        metadata = {
            "speaker_a": conversation.get("speaker_a", ""),
            "speaker_b": conversation.get("speaker_b", ""),
            "event_summary": raw_record.get("event_summary", {}),
            "session_summary": raw_record.get("session_summary", {}),
            "observation": raw_record.get("observation", {}),
            "question_count": len(qa_pairs),
            "session_count": len(sessions),
            "turn_ref_to_turn_id": dia_to_turn_id,
        }
        return DatasetDialogueExample(
            sample_id=sample_id,
            dataset_name=self.dataset_name,
            split=split,
            sessions=sessions,
            dialogue=dialogue,
            qa_pairs=qa_pairs,
            metadata=metadata,
        )

    @staticmethod
    def _session_from_dia_ref(dia_ref: str) -> str | None:
        if ":" not in dia_ref:
            return None
        prefix = dia_ref.split(":", 1)[0]
        if prefix.startswith("D") and prefix[1:].isdigit():
            return f"session_{prefix[1:]}"
        return None

    @staticmethod
    def _build_dia_to_turn_map(raw_sessions: list[list[Any]], sessions) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for raw_session, session in zip(raw_sessions, sessions):
            for raw_turn, turn in zip(raw_session, session.turns):
                dia_id = raw_turn.get("dia_id") if isinstance(raw_turn, dict) else None
                if dia_id:
                    mapping[str(dia_id)] = turn.turn_id
        return mapping

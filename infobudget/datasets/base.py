"""功能：定义统一数据集预处理与加载接口。
输入：原始数据记录、路径与 split。
输出：统一 DatasetDialogueExample。
依赖：abc、datetime、json、pathlib、schemas。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from infobudget.schemas import DatasetDialogueExample, DatasetQAPair, DatasetSession, Turn
from infobudget.utils.text import count_tokens


class BaseDatasetPreprocessor(ABC):
    """统一数据集预处理器基类。"""

    dataset_name: str

    @abstractmethod
    def normalize_record(
        self,
        raw_record: dict[str, Any],
        *,
        split: str,
        sample_index: int,
    ) -> DatasetDialogueExample:
        """将原始记录归一化为统一样本。"""

    def preprocess_file(self, raw_path: Path, output_path: Path, split: str) -> list[DatasetDialogueExample]:
        """处理单个原始文件。"""
        examples = list(self.iter_examples(iter_raw_records(raw_path), split=split))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
        return examples

    def preprocess_records(
        self,
        records: Iterable[dict[str, Any]],
        *,
        split: str,
    ) -> list[DatasetDialogueExample]:
        """将一组原始记录转换为统一样本。"""
        return list(self.iter_examples(records, split=split))

    def iter_examples(
        self,
        records: Iterable[dict[str, Any]],
        *,
        split: str,
    ) -> Iterator[DatasetDialogueExample]:
        """流式生成统一样本。"""
        for index, record in enumerate(records, start=1):
            yield self.normalize_record(record, split=split, sample_index=index)


def read_raw_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSON / JSONL 原始数据为列表。"""
    return list(iter_raw_records(path))


def iter_raw_records(path: Path) -> Iterator[dict[str, Any]]:
    """流式读取 JSON / JSONL 原始数据。"""
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
        return
    with path.open("r", encoding="utf-8") as handle:
        first_non_ws = handle.read(1)
        while first_non_ws and first_non_ws.isspace():
            first_non_ws = handle.read(1)
        if first_non_ws == "[":
            yield from _stream_json_array(handle)
            return
        handle.seek(0)
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ["data", "records", "examples", "items"]:
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload
        return
    raise ValueError(f"Unsupported raw dataset format: {path}")


def _stream_json_array(handle) -> Iterator[dict[str, Any]]:
    """流式解析顶层 JSON 数组中的对象。"""
    decoder = json.JSONDecoder()
    buffer = ""
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        buffer += chunk
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break
            if buffer[0] in ",]":
                buffer = buffer[1:]
                continue
            try:
                item, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            if isinstance(item, dict):
                yield item
            buffer = buffer[offset:]
    buffer = buffer.lstrip()
    while buffer and buffer[0] != "]":
        if buffer[0] == ",":
            buffer = buffer[1:].lstrip()
            continue
        item, offset = decoder.raw_decode(buffer)
        if isinstance(item, dict):
            yield item
        buffer = buffer[offset:].lstrip()


def normalize_turns(raw_turns: list[Any]) -> list[Turn]:
    """归一化对话轮次。"""
    turns: list[Turn] = []
    for index, item in enumerate(raw_turns, start=1):
        if isinstance(item, dict):
            raw_text = str(item.get("text") or item.get("content") or item.get("utterance") or "")
            caption = str(item.get("blip_caption") or "").strip()
            suffix = f" (image description: {caption})" if caption else ""
            text = raw_text if not caption or raw_text.endswith(suffix) else raw_text + suffix
            role = str(item.get("role") or item.get("speaker") or item.get("from") or "unknown")
            timestamp = item.get("timestamp")
            token_count = int(item.get("token_count") or count_tokens(text))
            metadata = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "text",
                    "content",
                    "utterance",
                    "role",
                    "speaker",
                    "from",
                    "timestamp",
                    "token_count",
                }
            }
            metadata.update(
                {
                    "raw_text": raw_text,
                    "segmentation_text": text,
                    "blip_caption": caption,
                    "image_description_appended": bool(caption),
                }
            )
        else:
            text = str(item)
            role = "unknown"
            timestamp = None
            token_count = count_tokens(text)
            metadata = {}
        turns.append(
            Turn(
                turn_id=index,
                role=role,
                text=text,
                token_count=token_count,
                timestamp=None if timestamp is None else str(timestamp),
                metadata=metadata,
            )
        )
    return turns


def normalize_qa_pairs(
    raw_qas: list[Any],
    sample_prefix: str,
    *,
    default_judge_profile: str = "generic",
    category_formatter: Callable[[Any], str] | None = None,
) -> list[DatasetQAPair]:
    """归一化问答对。"""
    pairs: list[DatasetQAPair] = []
    for index, item in enumerate(raw_qas, start=1):
        if isinstance(item, dict):
            question = str(item.get("question") or item.get("query") or item.get("prompt") or "")
            answer = str(item.get("answer") or item.get("gold_answer") or item.get("target") or "")
            evidence = item.get("evidence_turn_ids") or item.get("supporting_turn_ids") or []
            evidence_turn_refs = item.get("evidence") or item.get("supporting_turn_refs") or []
            evidence_session_ids = item.get("answer_session_ids") or item.get("evidence_session_ids") or []
            question_type = str(item.get("question_type") or "")
            category_raw = item.get("category", "")
            category = str(category_formatter(category_raw) if category_formatter else category_raw)
            question_date = item.get("question_date")
            is_unanswerable = bool(item.get("is_unanswerable") or str(item.get("question_id") or "").endswith("_abs"))
            judge_profile = str(item.get("judge_profile") or default_judge_profile)
            metadata = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "question",
                    "query",
                    "prompt",
                    "answer",
                    "gold_answer",
                    "target",
                    "evidence_turn_ids",
                    "supporting_turn_ids",
                    "evidence",
                    "supporting_turn_refs",
                    "answer_session_ids",
                    "evidence_session_ids",
                    "question_type",
                    "category",
                    "question_date",
                    "is_unanswerable",
                    "judge_profile",
                }
            }
        else:
            question = str(item)
            answer = ""
            evidence = []
            evidence_turn_refs = []
            evidence_session_ids = []
            question_type = ""
            category = ""
            question_date = None
            is_unanswerable = False
            judge_profile = default_judge_profile
            metadata = {}
        pairs.append(
            DatasetQAPair(
                question_id=f"{sample_prefix}_q{index:03d}",
                question=question,
                answer=answer,
                question_type=question_type,
                category=category,
                question_date=None if question_date is None else str(question_date),
                evidence_turn_ids=[int(value) for value in evidence if str(value).isdigit()],
                evidence_turn_refs=[str(value) for value in evidence_turn_refs],
                evidence_session_ids=[str(value) for value in evidence_session_ids],
                judge_profile=judge_profile,
                is_unanswerable=is_unanswerable,
                metadata=metadata,
            )
        )
    return pairs


def build_sessions_from_flat_turns(
    raw_sessions: list[list[Any]],
    *,
    session_ids: list[str] | None = None,
    raw_timestamps: list[str | None] | None = None,
    parser: Callable[[str | None], str | None] | None = None,
    synthesize_turn_timestamps: bool = False,
    turn_timestamp_step_ms: int = 1000,
) -> tuple[list[DatasetSession], list[Turn]]:
    """根据原始 session 列表构建 sessions 与扁平 dialogue。"""
    sessions: list[DatasetSession] = []
    dialogue: list[Turn] = []
    next_turn_id = 1
    for index, raw_session in enumerate(raw_sessions, start=1):
        session_id = session_ids[index - 1] if session_ids and index - 1 < len(session_ids) else f"session_{index}"
        raw_timestamp = raw_timestamps[index - 1] if raw_timestamps and index - 1 < len(raw_timestamps) else None
        parsed_timestamp = parse_optional_timestamp(raw_timestamp, parser)
        base_dt = _parse_normalized_datetime(parsed_timestamp)
        turns = normalize_turns(raw_session)
        for turn_offset, turn in enumerate(turns):
            # Some datasets only provide session-level timestamps. When requested,
            # synthesize turn-level times by applying a fixed offset within the session.
            if turn.timestamp is None:
                if synthesize_turn_timestamps and base_dt is not None:
                    turn_dt = base_dt + timedelta(milliseconds=turn_offset * turn_timestamp_step_ms)
                    turn.timestamp = turn_dt.isoformat(timespec="milliseconds")
                    turn.metadata.setdefault("weekday", turn_dt.strftime("%a"))
                    turn.metadata.setdefault("timestamp_source", "synthetic_turn_from_session")
                    turn.metadata.setdefault("timestamp_offset_ms", turn_offset * turn_timestamp_step_ms)
                else:
                    turn.timestamp = parsed_timestamp
                    if base_dt is not None:
                        turn.metadata.setdefault("weekday", base_dt.strftime("%a"))
                    turn.metadata.setdefault("timestamp_source", "session")
            turn.metadata.setdefault("session_id", session_id)
            turn.metadata.setdefault("session_timestamp", parsed_timestamp)
            turn.metadata.setdefault("session_raw_timestamp", raw_timestamp)
            turn.turn_id = next_turn_id
            turn.metadata.setdefault("display_turn_index", next_turn_id - 1)
            turn.metadata.setdefault("session_turn_index", turn_offset)
            turn.metadata.setdefault("segmentation_text", turn.text)
            next_turn_id += 1
        sessions.append(
            DatasetSession(
                session_id=session_id,
                timestamp=parsed_timestamp,
                raw_timestamp=raw_timestamp,
                turns=turns,
                metadata={},
            )
        )
        dialogue.extend(turns)
    return sessions, dialogue


def parse_optional_timestamp(
    raw_value: str | None,
    parser: Callable[[str | None], str | None] | None = None,
) -> str | None:
    """解析可选时间戳。"""
    if not raw_value:
        return None
    if parser is not None:
        parsed = parser(str(raw_value))
        return parsed if parsed is not None else str(raw_value)
    return str(raw_value)


def _parse_normalized_datetime(value: str | None) -> datetime | None:
    """Parse normalized timestamps produced by dataset parsers."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_locomo_timestamp(raw_value: str | None) -> str | None:
    """解析 LoCoMo 时间戳。"""
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, "%I:%M %p on %d %B, %Y").isoformat(sep=" ")
    except ValueError:
        return str(raw_value)


def parse_longmemeval_timestamp(raw_value: str | None) -> str | None:
    """解析 LongMemEval 时间戳。"""
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, "%Y/%m/%d (%a) %H:%M").isoformat(sep=" ")
    except ValueError:
        return str(raw_value)

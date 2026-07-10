"""功能：实现 JSONL + 向量索引长期记忆库。
输入：MemoryEntry、Episode 与查询向量。
输出：持久化文件与检索结果。
依赖：json、pathlib、numpy、schemas、向量索引。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from infobudget.config import StorageConfig
from infobudget.memory.qdrant_index import QdrantVectorIndex
from infobudget.schemas import MemoryEntry, Segment
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MemoryStore:
    """默认长期记忆存储。"""

    cfg: StorageConfig
    root_dir: Path
    entries: list[MemoryEntry] = field(default_factory=list)
    episode_entries: list[dict] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    jsonl_dir: Path = field(init=False)
    qdrant_dir: Path = field(init=False)
    memory_path: Path = field(init=False)
    episode_path: Path = field(init=False)
    segment_path: Path = field(init=False)
    memory_index: QdrantVectorIndex = field(init=False)
    episode_index: QdrantVectorIndex = field(init=False)

    def __post_init__(self) -> None:
        self.jsonl_dir = (self.root_dir / self.cfg.jsonl_dir).resolve()
        self.qdrant_dir = (self.root_dir / self.cfg.qdrant_dir).resolve()
        self.memory_path = self.jsonl_dir / "memory_entries.jsonl"
        self.episode_path = self.jsonl_dir / "episode_entries.jsonl"
        self.segment_path = self.jsonl_dir / "segments.jsonl"
        self.memory_index = QdrantVectorIndex(self.qdrant_dir, self.cfg.qdrant_memory_collection)
        self.episode_index = QdrantVectorIndex(self.qdrant_dir, self.cfg.qdrant_episode_collection)

    def add_entry(self, entry: MemoryEntry, embedding) -> str:
        self.entries.append(entry)
        self.memory_index.add(entry.id, embedding, entry.to_dict())
        return entry.id

    def add_episode(self, episode_entry: dict, embedding) -> str:
        episode_id = f"epi_{len(self.episode_entries)+1:06d}"
        payload = {"episode_id": episode_id, **episode_entry}
        self.episode_entries.append(payload)
        self.episode_index.add(episode_id, embedding, payload)
        return episode_id

    def record_segments(self, segments: list[Segment]) -> None:
        self.segments = list(segments)

    def retrieve(self, query_embedding, top_k: int = 5) -> list[MemoryEntry]:
        return [entry for entry, _score in self.search_by_embedding(query_embedding, top_k)]

    def search_by_embedding(self, query_embedding, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        payload_hits = self.memory_index.search_with_payload(query_embedding, top_k)
        if payload_hits:
            return [(self._memory_from_dict(payload), score) for payload, score in payload_hits]

        hits = self.memory_index.search(query_embedding, top_k)
        by_id = {entry.memory_id: entry for entry in self.entries}
        return [(by_id[item_id], score) for item_id, score in hits if item_id in by_id]

    def is_empty(self) -> bool:
        return len(self.entries) == 0 and self.memory_index.is_empty()

    def needs_index_rebuild(self) -> bool:
        """Return true when JSONL memories and Qdrant points are out of sync."""
        return bool(self.entries) and self.memory_index.count() != len(self.entries)

    def rebuild_indexes(self, encoder) -> None:
        """Rebuild Qdrant indexes from JSONL-backed entries."""
        self.memory_index.reset()
        self.episode_index.reset()
        for entry in self.entries:
            self.memory_index.add(entry.id, encoder.encode_text(entry.memory), entry.to_dict())
        logger.info("Memory indexes rebuilt from %s JSONL entries", len(self.entries))

    def save(self) -> None:
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(self.memory_path, [entry.to_dict() for entry in self.entries])
        self._write_jsonl(self.episode_path, self.episode_entries)
        self._write_jsonl(self.segment_path, [asdict(segment) for segment in self.segments])
        self.memory_index.save()
        self.episode_index.save()
        logger.info("Memory store saved %s entries", len(self.entries))

    def load(self) -> None:
        self.entries = [self._memory_from_dict(item) for item in self._read_jsonl(self.memory_path)]
        self.episode_entries = self._read_jsonl(self.episode_path)
        self.segments = [Segment(**item) for item in self._read_jsonl(self.segment_path)]
        self.memory_index.load()
        self.episode_index.load()

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _memory_from_dict(raw: dict) -> MemoryEntry:
        return MemoryEntry(
            id=raw.get("id", raw.get("memory_id", "")),
            time_stamp=raw.get("time_stamp", ""),
            float_time_stamp=float(raw.get("float_time_stamp", 0.0) or 0.0),
            weekday=raw.get("weekday", ""),
            topic_id=int(raw.get("topic_id", 0) or 0),
            topic_summary=raw.get("topic_summary", ""),
            memory=raw.get("memory", raw.get("summary", "")),
            original_memory=raw.get("original_memory", ""),
            compressed_memory=raw.get("compressed_memory", ""),
            entry_type=raw.get("entry_type", "factual"),
            speaker_id=raw.get("speaker_id", "unknown"),
            speaker_name=raw.get("speaker_name", "User"),
            consolidated=bool(raw.get("consolidated", False)),
            update_queue=raw.get("update_queue", []),
            source_segment_id=raw.get("source_segment_id", ""),
            source_turn_id=int(raw.get("source_turn_id", 0) or 0),
            source_turn_ids=[int(item) for item in raw.get("source_turn_ids", []) if str(item).isdigit()],
            source_start_turn=int(raw.get("source_start_turn", 0) or 0),
            source_end_turn=int(raw.get("source_end_turn", 0) or 0),
        )

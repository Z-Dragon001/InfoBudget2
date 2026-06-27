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

import numpy as np

from infobudget.config import StorageConfig
from infobudget.memory.vector_index import BaseVectorIndex, NumpyFlatIPIndex
from infobudget.schemas import (
    Constraint,
    Episode,
    EpisodicMemory,
    MemoryEntry,
    Preference,
    ScoreResult,
    Segment,
    SemanticEntity,
    SemanticFact,
    SemanticMemory,
)
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MemoryStore:
    """默认长期记忆存储。"""

    cfg: StorageConfig
    root_dir: Path
    memory_index: BaseVectorIndex = field(default_factory=NumpyFlatIPIndex)
    episode_index: BaseVectorIndex = field(default_factory=NumpyFlatIPIndex)
    entries: list[MemoryEntry] = field(default_factory=list)
    episode_entries: list[dict] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    jsonl_dir: Path = field(init=False)
    faiss_dir: Path = field(init=False)
    memory_path: Path = field(init=False)
    episode_path: Path = field(init=False)
    segment_path: Path = field(init=False)
    memory_index_path: Path = field(init=False)
    episode_index_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.jsonl_dir = (self.root_dir / self.cfg.jsonl_dir).resolve()
        self.faiss_dir = (self.root_dir / self.cfg.faiss_dir).resolve()
        self.memory_path = self.jsonl_dir / "memory_entries.jsonl"
        self.episode_path = self.jsonl_dir / "episode_entries.jsonl"
        self.segment_path = self.jsonl_dir / "segments.jsonl"
        self.memory_index_path = self.faiss_dir / "memory.index"
        self.episode_index_path = self.faiss_dir / "episode.index"

    def add_entry(self, entry: MemoryEntry, embedding: np.ndarray) -> str:
        entry.memory_id = f"mem_{len(self.entries)+1:06d}"
        entry.embedding_id = len(self.entries)
        self.entries.append(entry)
        self.memory_index.add(entry.memory_id, embedding)
        return entry.memory_id

    def add_episode(self, episode_entry: dict, embedding: np.ndarray) -> str:
        episode_id = f"epi_{len(self.episode_entries)+1:06d}"
        payload = {"episode_id": episode_id, **episode_entry}
        self.episode_entries.append(payload)
        self.episode_index.add(episode_id, embedding)
        return episode_id

    def record_segments(self, segments: list[Segment]) -> None:
        self.segments = list(segments)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> list[MemoryEntry]:
        return [entry for entry, _score in self.search_by_embedding(query_embedding, top_k)]

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        hits = self.memory_index.search(query_embedding, top_k)
        by_id = {entry.memory_id: entry for entry in self.entries}
        return [(by_id[item_id], score) for item_id, score in hits if item_id in by_id]

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def save(self) -> None:
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(self.memory_path, [entry.to_dict() for entry in self.entries])
        self._write_jsonl(self.episode_path, self.episode_entries)
        self._write_jsonl(self.segment_path, [asdict(segment) for segment in self.segments])
        self.memory_index.save(self.memory_index_path)
        self.episode_index.save(self.episode_index_path)
        logger.info("Memory store saved %s entries", len(self.entries))

    def load(self) -> None:
        self.entries = [self._memory_from_dict(item) for item in self._read_jsonl(self.memory_path)]
        self.episode_entries = self._read_jsonl(self.episode_path)
        self.segments = [Segment(**item) for item in self._read_jsonl(self.segment_path)]
        self.memory_index.load(self.memory_index_path)
        self.episode_index.load(self.episode_index_path)

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
        semantic = raw["semantic_memory"]
        episodic = raw["episodic_memory"]
        return MemoryEntry(
            memory_id=raw["memory_id"],
            segment_id=raw["segment_id"],
            topic=raw["topic"],
            summary=raw["summary"],
            semantic_memory=SemanticMemory(
                entities=[SemanticEntity(**item) for item in semantic["entities"]],
                facts=[SemanticFact(**item) for item in semantic["facts"]],
                preferences=[Preference(**item) for item in semantic["preferences"]],
                constraints=[Constraint(**item) for item in semantic["constraints"]],
            ),
            episodic_memory=EpisodicMemory(
                episodes=[Episode(**item) for item in episodic["episodes"]]
            ),
            importance=raw["importance"],
            information_score=raw["information_score"],
            router_level=raw["router_level"],
            extraction_mode=raw["extraction_mode"],
            extractor_name=raw["extractor_name"],
            model_used=raw["model_used"],
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            latency_ms=raw["latency_ms"],
            cost_usd=raw["cost_usd"],
            created_at=raw["created_at"],
            embedding_id=raw["embedding_id"],
        )

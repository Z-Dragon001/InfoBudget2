"""Tests for LightMem-compatible memory store persistence and retrieval."""

from pathlib import Path
import json
import shutil

from infobudget.config import load_project_bundle
from infobudget.memory.store import MemoryStore
from infobudget.schemas import MemoryEntry
from infobudget.utils.embeddings import HashingTextEncoder


def test_memory_store_roundtrip(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    store = MemoryStore(bundle.config.storage, tmp_path)
    encoder = HashingTextEncoder()
    entry = MemoryEntry(
        time_stamp="2025-01-10T00:00:00.000",
        float_time_stamp=1736467200.0,
        weekday="Fri",
        topic_id=0,
        topic_summary="",
        memory="InfoBudget focuses on long-term memory construction cost.",
        original_memory="",
        compressed_memory="",
        speaker_id="unknown",
        speaker_name="User",
        consolidated=False,
        update_queue=[],
    )

    store.add_entry(entry, encoder.encode_text(entry.memory))
    store.save()
    loaded = MemoryStore(bundle.config.storage, tmp_path)
    loaded.load()
    hits = loaded.retrieve(encoder.encode_text("InfoBudget cost"), 1)

    assert len(hits) == 1
    assert hits[0].topic_id == 0
    assert hits[0].memory == "InfoBudget focuses on long-term memory construction cost."


def test_memory_store_can_rebuild_qdrant_from_jsonl(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    encoder = HashingTextEncoder()
    store = MemoryStore(bundle.config.storage, tmp_path)
    entry = MemoryEntry(memory="JSONL can restore a missing Qdrant index.")

    store.add_entry(entry, encoder.encode_text(entry.memory))
    store.save()
    shutil.rmtree(tmp_path / "outputs" / "qdrant")

    loaded = MemoryStore(bundle.config.storage, tmp_path)
    loaded.load()
    assert loaded.needs_index_rebuild()

    loaded.rebuild_indexes(encoder)
    loaded.save()
    hits = loaded.retrieve(encoder.encode_text("restore Qdrant index"), 1)

    assert len(hits) == 1
    assert hits[0].memory == "JSONL can restore a missing Qdrant index."


def test_memory_store_retrieval_prefers_qdrant_payload_over_jsonl(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    encoder = HashingTextEncoder()
    store = MemoryStore(bundle.config.storage, tmp_path)
    entry = MemoryEntry(
        memory="Qdrant payload is the QA retrieval source.",
        original_memory="Qdrant payload is the QA retrieval source.",
        source_segment_id="seg_000123",
        source_turn_id=7,
        source_turn_ids=[7, 8],
        source_start_turn=7,
        source_end_turn=8,
    )

    store.add_entry(entry, encoder.encode_text(entry.memory))
    store.save()
    stale_row = entry.to_dict()
    stale_row["memory"] = "JSONL is only for human inspection."
    memory_path = tmp_path / "outputs" / "memory_jsonl" / "memory_entries.jsonl"
    memory_path.write_text(json.dumps(stale_row, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = MemoryStore(bundle.config.storage, tmp_path)
    loaded.load()
    hits = loaded.retrieve(encoder.encode_text("QA retrieval source"), 1)

    assert len(hits) == 1
    assert hits[0].memory == "Qdrant payload is the QA retrieval source."
    assert hits[0].segment_id == "seg_000123"
    assert hits[0].source_turn_ids == [7, 8]

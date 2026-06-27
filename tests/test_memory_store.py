"""功能：测试 MemoryStore 保存与检索。
输入：模拟 memory entry。
输出：持久化与加载断言。
依赖：pytest、项目存储模块。
作者：OpenAI Codex
"""

from pathlib import Path

from infobudget.config import load_project_bundle
from infobudget.memory.store import MemoryStore
from infobudget.schemas import EpisodicMemory, MemoryEntry, SemanticMemory
from infobudget.utils.embeddings import HashingTextEncoder


def test_memory_store_roundtrip(tmp_path: Path) -> None:
    bundle = load_project_bundle("configs")
    store = MemoryStore(bundle.config.storage, tmp_path)
    encoder = HashingTextEncoder()
    entry = MemoryEntry(
        memory_id="",
        segment_id="seg_000001",
        topic="infobudget",
        summary="InfoBudget 关注长期记忆构建成本。",
        semantic_memory=SemanticMemory(),
        episodic_memory=EpisodicMemory(),
        importance=0.7,
        information_score=0.6,
        router_level="medium",
        extraction_mode="joint",
        extractor_name="mock_joint_extractor",
        model_used="gpt-4.1-mini",
        input_tokens=100,
        output_tokens=50,
        latency_ms=120,
        cost_usd=0.001,
    )
    store.add_entry(entry, encoder.encode_text(entry.summary))
    store.save()
    loaded = MemoryStore(bundle.config.storage, tmp_path)
    loaded.load()
    hits = loaded.retrieve(encoder.encode_text("InfoBudget 成本"), 1)
    assert len(hits) == 1
    assert hits[0].topic == "infobudget"

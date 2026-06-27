"""功能：实现基于 summary embedding 的长期记忆检索。
输入：query 文本。
输出：相关 MemoryEntry 列表。
依赖：memory、embeddings。
作者：OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass

from infobudget.memory.store import MemoryStore
from infobudget.schemas import MemoryEntry
from infobudget.utils.embeddings import HashingTextEncoder


@dataclass(slots=True)
class Retriever:
    """简单检索器。"""

    store: MemoryStore
    encoder: HashingTextEncoder

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        embedding = self.encoder.encode_text(query)
        return self.store.retrieve(embedding, top_k)

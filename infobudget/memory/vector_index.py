"""功能：定义向量索引接口并提供 numpy / FAISS 实现。
输入：向量与条目标识。
输出：相似度检索结果与索引文件。
依赖：abc、pickle、numpy。
作者：OpenAI Codex
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from infobudget.utils.embeddings import cosine_similarity

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


class BaseVectorIndex(ABC):
    """向量索引基类。"""

    @abstractmethod
    def add(self, item_id: str, vector: np.ndarray) -> None:
        """添加向量。"""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """检索向量。"""

    @abstractmethod
    def save(self, path: Path) -> None:
        """保存索引。"""

    @abstractmethod
    def load(self, path: Path) -> None:
        """加载索引。"""

    @abstractmethod
    def is_empty(self) -> bool:
        """判断是否为空。"""


class NumpyFlatIPIndex(BaseVectorIndex):
    """基于 numpy 的扁平内积索引。"""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors: list[np.ndarray] = []

    def add(self, item_id: str, vector: np.ndarray) -> None:
        self.ids.append(item_id)
        self.vectors.append(vector.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self.is_empty():
            return []
        if query_vector.ndim != 1 or query_vector.size != self.vectors[0].size:
            query_vector = self.vectors[-1]
        scored = [(item_id, cosine_similarity(query_vector, vector)) for item_id, vector in zip(self.ids, self.vectors)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ids": self.ids, "vectors": self.vectors}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self.ids = payload.get("ids", [])
        self.vectors = payload.get("vectors", [])

    def is_empty(self) -> bool:
        return not self.ids


class FaissVectorIndex(BaseVectorIndex):
    """可选的 FAISS FlatIP 索引。"""

    def __init__(self, dim: int) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, item_id: str, vector: np.ndarray) -> None:
        self.ids.append(item_id)
        self.index.add(vector.astype(np.float32).reshape(1, -1))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self.is_empty():
            return []
        scores, indexes = self.index.search(query_vector.astype(np.float32).reshape(1, -1), top_k)
        results: list[tuple[str, float]] = []
        for score, index in zip(scores[0], indexes[0]):
            if index >= 0:
                results.append((self.ids[index], float(score)))
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        with path.with_suffix(path.suffix + ".ids").open("wb") as handle:
            pickle.dump(self.ids, handle)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        self.index = faiss.read_index(str(path))
        with path.with_suffix(path.suffix + ".ids").open("rb") as handle:
            self.ids = pickle.load(handle)

    def is_empty(self) -> bool:
        return len(self.ids) == 0

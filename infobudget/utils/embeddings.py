"""功能：提供无外部模型依赖的轻量文本向量编码。
输入：文本或文本列表。
输出：L2 归一化向量。
依赖：hashlib、numpy。
作者：OpenAI Codex
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from infobudget.utils.text import tokenize_text


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """执行 L2 归一化。"""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


@dataclass(slots=True)
class HashingTextEncoder:
    """基于哈希特征的轻量编码器。"""

    dim: int = 256

    def _index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.dim

    def encode_text(self, text: str) -> np.ndarray:
        """编码单条文本。"""
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize_text(text)
        if not tokens:
            return vector
        for token in tokens:
            vector[self._index(token)] += 1.0
        for left, right in zip(tokens, tokens[1:]):
            vector[self._index(f"{left}::{right}")] += 0.5
        return _l2_normalize(vector)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """编码文本列表。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.encode_text(text) for text in texts]).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """计算余弦相似度。"""
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))

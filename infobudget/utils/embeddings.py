"""功能：提供文本向量编码器。
输入：文本或文本列表。
输出：L2 归一化向量。
依赖：hashlib、numpy、可选 sentence-transformers。
作者：OpenAI Codex
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from infobudget.utils.text import tokenize_text
from infobudget.utils.logging import get_logger

logger = get_logger(__name__)


class TextEncoder(Protocol):
    """Text encoder interface used by segmentation, scoring, and storage."""

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one text string."""

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of text strings."""


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


@dataclass(slots=True)
class SentenceTransformerTextEncoder:
    """Sentence-Transformers encoder with a hashing fallback for offline runs."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    fallback_on_error: bool = True
    fallback_encoder: HashingTextEncoder | None = None
    _model: object | None = field(default=None, init=False, repr=False)
    _disabled_reason: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fallback_encoder is None:
            self.fallback_encoder = HashingTextEncoder(dim=self.dim)

    def encode_text(self, text: str) -> np.ndarray:
        """编码单条文本。"""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """编码文本列表。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load_model()
        if model is None:
            return self._fallback_encoder().encode_batch(texts)
        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            if not self.fallback_on_error:
                raise
            self._disabled_reason = str(exc)
            logger.warning(
                "SentenceTransformer encoding failed for %s; falling back to hashing encoder: %s",
                self.model_name,
                exc,
            )
            return self._fallback_encoder().encode_batch(texts)
        return np.asarray(embeddings, dtype=np.float32)

    def _fallback_encoder(self) -> HashingTextEncoder:
        if self.fallback_encoder is None:
            self.fallback_encoder = HashingTextEncoder(dim=self.dim)
        return self.fallback_encoder

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._disabled_reason:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            if not self.fallback_on_error:
                raise
            self._disabled_reason = str(exc)
            logger.warning(
                "sentence-transformers is not installed; falling back to hashing encoder for %s",
                self.model_name,
            )
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension:
                self.dim = int(dimension)
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            if not self.fallback_on_error:
                raise
            self._disabled_reason = str(exc)
            logger.warning(
                "Could not load SentenceTransformer model %s; falling back to hashing encoder: %s",
                self.model_name,
                exc,
            )
            return None
        return self._model


def build_text_encoder(model_name: str) -> TextEncoder:
    """Create a text encoder from a config model name."""
    normalized = (model_name or "").strip()
    if not normalized or normalized == "hashing-encoder-v1":
        return HashingTextEncoder()
    if normalized in {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}:
        return SentenceTransformerTextEncoder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    if normalized.startswith("sentence-transformers/"):
        return SentenceTransformerTextEncoder(normalized)
    logger.warning("Unknown embedding model %s; falling back to hashing encoder", normalized)
    return HashingTextEncoder()


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """计算余弦相似度。"""
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))

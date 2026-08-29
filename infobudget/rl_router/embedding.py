"""Strict local embedding and tokenizer adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalSentenceEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        local_path: str | Path,
        dimension: int,
        normalize: bool = True,
        max_length: int | None = None,
        long_text_strategy: str = "truncate",
    ):
        self.model_name = model_name
        self.local_path = Path(local_path).resolve()
        self.dimension = int(dimension)
        self.normalize = normalize
        self.max_length = int(max_length) if max_length is not None else None
        self.long_text_strategy = str(long_text_strategy).strip().lower()
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("embedding max_length must be positive")
        if self.long_text_strategy not in {"truncate", "mean_pool_chunks"}:
            raise ValueError(
                "embedding long_text_strategy must be truncate or mean_pool_chunks"
            )
        if not self.local_path.is_dir():
            raise FileNotFoundError(f"local embedding model is missing: {self.local_path}")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(self.local_path), local_files_only=True)
        if self.max_length is not None:
            self._model.max_seq_length = self.max_length
        get_dimension = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)
        actual = int(get_dimension())
        if actual != self.dimension:
            raise ValueError(f"embedding dimension mismatch: configured={self.dimension}, actual={actual}")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.long_text_strategy == "mean_pool_chunks":
            values = self._encode_chunk_means(texts)
        else:
            values = self._model.encode(
                texts,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(f"unexpected embedding shape: {array.shape}")
        return array

    def _encode_chunk_means(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.max_length is None:
            raise ValueError("mean_pool_chunks requires max_length")
        tokenizer = self._model.tokenizer
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        chunk_size = self.max_length - special_tokens
        if chunk_size <= 0:
            raise ValueError("embedding max_length leaves no room for text tokens")
        chunks: list[str] = []
        owners: list[int] = []
        for owner, text in enumerate(texts):
            token_ids = tokenizer.encode(
                str(text),
                add_special_tokens=False,
                truncation=False,
                verbose=False,
            )
            token_chunks = [
                token_ids[start : start + chunk_size]
                for start in range(0, len(token_ids), chunk_size)
            ] or [[]]
            chunks.extend(
                tokenizer.decode(
                    item,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for item in token_chunks
            )
            owners.extend([owner] * len(token_chunks))
        chunk_vectors = np.asarray(
            self._model.encode(
                chunks,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        pooled = np.zeros((len(texts), self.dimension), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        for owner, vector in zip(owners, chunk_vectors):
            pooled[owner] += vector
            counts[owner] += 1.0
        pooled /= counts[:, None]
        if self.normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled /= np.maximum(norms, 1e-12)
        return pooled


class LocalTokenizer:
    def __init__(self, path: str | Path):
        local_path = Path(path).resolve()
        if not local_path.is_dir():
            raise FileNotFoundError(f"local tokenizer is missing: {local_path}")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def directory_hash(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for file in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(file.relative_to(root).as_posix().encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

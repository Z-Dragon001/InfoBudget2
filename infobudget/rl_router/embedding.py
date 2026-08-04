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
    def __init__(self, *, model_name: str, local_path: str | Path, dimension: int, normalize: bool = True):
        self.model_name = model_name
        self.local_path = Path(local_path).resolve()
        self.dimension = int(dimension)
        self.normalize = normalize
        if not self.local_path.is_dir():
            raise FileNotFoundError(f"local embedding model is missing: {self.local_path}")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(str(self.local_path), local_files_only=True)
        get_dimension = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)
        actual = int(get_dimension())
        if actual != self.dimension:
            raise ValueError(f"embedding dimension mismatch: configured={self.dimension}, actual={actual}")

    def encode(self, texts: list[str]) -> np.ndarray:
        values = self._model.encode(texts, normalize_embeddings=self.normalize, convert_to_numpy=True)
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise ValueError(f"unexpected embedding shape: {array.shape}")
        return array


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

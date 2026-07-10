"""Qdrant-backed vector indexes for long-term memory retrieval."""

from __future__ import annotations

import uuid
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantVectorIndex:
    """Small wrapper around a local Qdrant collection.

    Qdrant point ids must be integers or UUIDs. InfoBudget memory ids are UUIDs,
    but episode ids can be arbitrary strings, so every external id is mapped to
    a stable UUID and the original id is kept in the payload as ``item_id``.
    """

    def __init__(self, path: Path, collection_name: str) -> None:
        self.path = path / collection_name
        self.collection_name = collection_name
        self.client: QdrantClient | None = None
        self.vector_size: int | None = None
        self.path.mkdir(parents=True, exist_ok=True)

    def add(self, item_id: str, vector: np.ndarray, payload: dict[str, Any] | None = None) -> None:
        vector = self._as_float_vector(vector)
        self._ensure_collection(len(vector))
        point_payload = {"item_id": item_id, **(payload or {})}
        self._client().upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self._point_id(item_id),
                    vector=vector,
                    payload=point_payload,
                )
            ],
            wait=True,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Return external item ids and similarity scores for a query vector."""
        return [
            (str(payload.get("item_id", "")), score)
            for payload, score in self.search_with_payload(query_vector, top_k)
        ]

    def search_with_payload(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[dict[str, Any], float]]:
        """Return Qdrant payloads and similarity scores for a query vector."""
        if self.is_empty():
            return []
        query = self._as_float_vector(query_vector)
        response = self._client().query_points(
            collection_name=self.collection_name,
            query=query,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results: list[tuple[dict[str, Any], float]] = []
        for point in response.points:
            payload = point.payload or {}
            if "item_id" not in payload:
                payload = {"item_id": str(point.id), **payload}
            results.append((payload, float(point.score)))
        return results

    def save(self) -> None:
        """Flush and release the local Qdrant lock."""
        self.close()

    def load(self) -> None:
        if self._collection_exists():
            config = self._client().get_collection(self.collection_name).config
            params = config.params.vectors
            if isinstance(params, models.VectorParams):
                self.vector_size = int(params.size)

    def is_empty(self) -> bool:
        return self.count() == 0

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        count = self._client().count(collection_name=self.collection_name, exact=True)
        return int(count.count)

    def reset(self) -> None:
        """Drop the local collection storage for a clean rebuild."""
        self.close()
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.vector_size = None

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def _client(self) -> QdrantClient:
        if self.client is None:
            self.client = QdrantClient(path=str(self.path))
        return self.client

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            if self.vector_size is None:
                self.load()
            return
        self._client().create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        self.vector_size = vector_size

    def _collection_exists(self) -> bool:
        return self._client().collection_exists(self.collection_name)

    @staticmethod
    def _as_float_vector(vector: np.ndarray) -> list[float]:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        return [float(item) for item in array]

    @staticmethod
    def _point_id(item_id: str) -> str:
        try:
            return str(uuid.UUID(item_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"infobudget:{item_id}"))

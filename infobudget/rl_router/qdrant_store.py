"""Filter-safe Qdrant storage for immutable L/M/H candidates and S assemblies."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient, models

from infobudget.rl_router.schemas import FactRecord, Tier, TopicSegment

COLLECTION_SUFFIX = {"small": "L", "medium": "M", "large": "H", "assembled": "S"}


@dataclass(slots=True)
class AssemblyResult:
    assembly_id: str
    sample_id: str
    status: str
    point_count: int
    route: dict[str, Tier]


class FactQdrantStore:
    def __init__(
        self,
        path: str | Path | None,
        namespace: str,
        vector_size: int,
        *,
        in_memory: bool = False,
        url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        prefer_grpc: bool = False,
        grpc_port: int = 6334,
        distance: str = "Cosine",
        read_only: bool = False,
    ):
        if not namespace or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in namespace):
            raise ValueError("namespace may contain only letters, digits, '_' and '-'")
        if in_memory and url:
            raise ValueError("in_memory and url are mutually exclusive")
        if not in_memory and not url and path is None:
            raise ValueError("local Qdrant requires a storage path")
        self.namespace = namespace
        self.read_only = bool(read_only)
        self.vector_size = int(vector_size)
        try:
            self.distance = {
                "COSINE": models.Distance.COSINE,
                "DOT": models.Distance.DOT,
                "EUCLID": models.Distance.EUCLID,
                "MANHATTAN": models.Distance.MANHATTAN,
            }[str(distance).strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unsupported Qdrant distance: {distance!r}") from exc
        self.collections = {name: f"{namespace}_{suffix}" for name, suffix in COLLECTION_SUFFIX.items()}
        self.candidates_frozen = False
        if in_memory:
            self.mode = "memory"
            self.client = QdrantClient(":memory:")
        elif url:
            self.mode = "server"
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=float(timeout_seconds),
                prefer_grpc=bool(prefer_grpc),
                grpc_port=int(grpc_port),
            )
            try:
                self.client.get_collections()
            except Exception as exc:
                self.client.close()
                raise ConnectionError(
                    f"cannot connect to Qdrant server at {url}; start the server and check storage.url"
                ) from exc
        else:
            self.mode = "local"
            self.client = QdrantClient(path=str(Path(path).resolve()))
        self._create_collections()

    @classmethod
    def from_storage_config(
        cls,
        storage: dict[str, Any],
        *,
        project_root: str | Path,
        namespace: str,
        read_only: bool = False,
    ) -> "FactQdrantStore":
        """Create the configured formal store without leaking credentials into config files."""
        mode = str(storage.get("mode", "")).strip().lower()
        vector_size = int(storage["vector_size"])
        if mode == "server":
            api_key_env = str(storage.get("api_key_env") or "").strip()
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if api_key_env and not api_key:
                raise EnvironmentError(
                    f"Qdrant server API key environment variable is missing: {api_key_env}"
                )
            return cls(
                None,
                namespace,
                vector_size,
                url=str(storage["url"]),
                api_key=api_key,
                timeout_seconds=float(storage.get("timeout_seconds", 30)),
                prefer_grpc=bool(storage.get("prefer_grpc", False)),
                grpc_port=int(storage.get("grpc_port", 6334)),
                distance=str(storage.get("distance", "Cosine")),
                read_only=read_only,
            )
        if mode == "local":
            local_path = Path(str(storage["local_path"]))
            if not local_path.is_absolute():
                local_path = Path(project_root) / local_path
            return cls(
                local_path,
                namespace,
                vector_size,
                distance=str(storage.get("distance", "Cosine")),
                read_only=read_only,
            )
        raise ValueError(f"unsupported Qdrant storage mode: {mode!r}")

    @staticmethod
    def probe_storage_config(storage: dict[str, Any]) -> None:
        """Fail fast on server credentials/connectivity without creating collections."""
        mode = str(storage.get("mode", "")).strip().lower()
        if mode == "local":
            return
        if mode != "server":
            raise ValueError(f"unsupported Qdrant storage mode: {mode!r}")
        api_key_env = str(storage.get("api_key_env") or "").strip()
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise EnvironmentError(
                f"Qdrant server API key environment variable is missing: {api_key_env}"
            )
        url = str(storage["url"])
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=float(storage.get("timeout_seconds", 30)),
            prefer_grpc=bool(storage.get("prefer_grpc", False)),
            grpc_port=int(storage.get("grpc_port", 6334)),
        )
        try:
            client.get_collections()
        except Exception as exc:
            raise ConnectionError(
                f"cannot connect to Qdrant server at {url}; start it before running the pipeline"
            ) from exc
        finally:
            client.close()

    def close(self) -> None:
        self.client.close()

    def upsert_facts(self, tier: Tier, facts: list[FactRecord], vectors: np.ndarray) -> None:
        self._require_writable()
        if self.candidates_frozen:
            raise RuntimeError("L/M/H candidates are frozen")
        if len(facts) != len(vectors):
            raise ValueError("one vector is required per fact")
        points = []
        for fact, vector in zip(facts, vectors):
            if fact.memory_tier != tier:
                raise ValueError("fact tier does not match target collection")
            points.append(models.PointStruct(id=fact.fact_id, vector=self._vector(vector), payload=fact.payload()))
        if points:
            self.client.upsert(self.collections[tier], points=points, wait=True)

    def replace_candidate_batch(
        self,
        tier: Tier,
        *,
        dataset_name: str,
        split: str,
        sample_id: str,
        extraction_run_id: str,
        batch_id: str,
        facts: list[FactRecord],
        vectors: np.ndarray,
    ) -> None:
        """Idempotently replace every candidate Point owned by one extraction batch."""
        self._require_writable()
        if self.candidates_frozen:
            raise RuntimeError("L/M/H candidates are frozen")
        if not extraction_run_id or not batch_id:
            raise ValueError("candidate batch replacement requires run_id and batch_id")
        if len(facts) != len(vectors):
            raise ValueError("one vector is required per fact")
        for fact in facts:
            actual = (
                fact.memory_tier,
                fact.dataset_name,
                fact.split,
                fact.sample_id,
                fact.extraction_run_id,
                fact.batch_id,
            )
            expected = (
                tier,
                dataset_name,
                split,
                sample_id,
                extraction_run_id,
                batch_id,
            )
            if actual != expected:
                raise ValueError("fact ownership does not match candidate batch replacement scope")
        conditions = [
            *self._sample_conditions(dataset_name, split, sample_id),
            self._match("extraction_run_id", extraction_run_id),
            self._match("batch_id", batch_id),
        ]
        self.client.delete(
            self.collections[tier],
            points_selector=models.FilterSelector(filter=models.Filter(must=conditions)),
            wait=True,
        )
        self.upsert_facts(tier, facts, vectors)

    def candidate_points(
        self,
        tier: Tier,
        *,
        dataset_name: str,
        split: str,
        sample_id: str,
        segment_id: str | None = None,
        extraction_run_id: str | None = None,
        batch_id: str | None = None,
        with_vectors: bool = True,
    ) -> list[Any]:
        conditions = self._sample_conditions(dataset_name, split, sample_id)
        if segment_id is not None:
            conditions.append(self._match("segment_id", segment_id))
        if extraction_run_id is not None:
            conditions.append(self._match("extraction_run_id", extraction_run_id))
        if batch_id is not None:
            conditions.append(self._match("batch_id", batch_id))
        return self._scroll(self.collections[tier], conditions, with_vectors=with_vectors)

    def assemble(
        self,
        *,
        dataset_name: str,
        split: str,
        sample_id: str,
        segments: list[TopicSegment],
        actions: list[Tier],
        episode_id: str,
        policy_version: str,
        extraction_run_id: str | None = None,
        assembly_id: str | None = None,
    ) -> AssemblyResult:
        self._require_writable()
        if len(segments) != len(actions) or not segments:
            raise ValueError("assembly requires one action per segment")
        if any(segment.sample_id != sample_id for segment in segments):
            raise ValueError("assembly contains a segment from another sample")
        assembly_id = assembly_id or str(uuid.uuid4())
        route = {segment.segment_id: tier for segment, tier in zip(segments, actions)}
        if len(route) != len(segments):
            raise ValueError("duplicate segment_id in assembly route")
        points: list[models.PointStruct] = []
        for segment, tier in zip(segments, actions):
            source = self.candidate_points(
                tier,
                dataset_name=dataset_name,
                split=split,
                sample_id=sample_id,
                segment_id=segment.segment_id,
                extraction_run_id=extraction_run_id,
                with_vectors=True,
            )
            for point in source:
                payload = dict(point.payload or {})
                payload.update(
                    {
                        "source_collection_tier": tier,
                        "source_point_id": str(point.id),
                        "source_extraction_run_id": payload.get("extraction_run_id", ""),
                        "assembly_id": assembly_id,
                        "episode_id": episode_id,
                        "policy_version": policy_version,
                    }
                )
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{assembly_id}:{tier}:{point.id}"))
                points.append(models.PointStruct(id=point_id, vector=point.vector, payload=payload))
        if points:
            self.client.upsert(self.collections["assembled"], points=points, wait=True)
        stored = self.assembly_points(
            dataset_name=dataset_name,
            split=split,
            sample_id=sample_id,
            assembly_id=assembly_id,
            with_vectors=False,
        )
        selected_by_segment: dict[str, set[str]] = {}
        for point in stored:
            payload = point.payload or {}
            selected_by_segment.setdefault(str(payload.get("segment_id")), set()).add(
                str(payload.get("source_collection_tier"))
            )
        valid = all(len(tiers) == 1 and route[segment_id] in tiers for segment_id, tiers in selected_by_segment.items())
        status = "ready" if valid and len(stored) == len(points) else "failed"
        if status == "failed":
            self.delete_assembly(dataset_name=dataset_name, split=split, sample_id=sample_id, assembly_id=assembly_id)
        return AssemblyResult(assembly_id, sample_id, status, len(stored), route)

    def assembly_points(self, *, dataset_name: str, split: str, sample_id: str, assembly_id: str, with_vectors: bool = False) -> list[Any]:
        if not assembly_id:
            raise ValueError("S collection operations require assembly_id")
        return self._scroll(
            self.collections["assembled"],
            [*self._sample_conditions(dataset_name, split, sample_id), self._match("assembly_id", assembly_id)],
            with_vectors=with_vectors,
        )

    def search_assembly(
        self,
        query_vector: np.ndarray,
        *,
        dataset_name: str,
        split: str,
        sample_id: str,
        assembly_id: str,
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        if not assembly_id:
            raise ValueError("S search requires assembly_id")
        query_filter = models.Filter(
            must=[*self._sample_conditions(dataset_name, split, sample_id), self._match("assembly_id", assembly_id)]
        )
        result = self.client.query_points(
            self.collections["assembled"],
            query=self._vector(query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [(dict(point.payload or {}), float(point.score)) for point in result.points]

    def delete_assembly(self, *, dataset_name: str, split: str, sample_id: str, assembly_id: str) -> None:
        self._require_writable()
        if not assembly_id:
            raise ValueError("S delete requires assembly_id")
        self.client.delete(
            self.collections["assembled"],
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[*self._sample_conditions(dataset_name, split, sample_id), self._match("assembly_id", assembly_id)]
                )
            ),
            wait=True,
        )

    def count_sample(self, collection: Tier | str, *, dataset_name: str, split: str, sample_id: str, assembly_id: str | None = None) -> int:
        if collection == "assembled" and not assembly_id:
            raise ValueError("S count requires assembly_id")
        conditions = self._sample_conditions(dataset_name, split, sample_id)
        if assembly_id:
            conditions.append(self._match("assembly_id", assembly_id))
        response = self.client.count(self.collections[collection], count_filter=models.Filter(must=conditions), exact=True)
        return int(response.count)

    def freeze_candidates(self) -> None:
        self.candidates_frozen = True

    def _create_collections(self) -> None:
        for name in self.collections.values():
            if not self.client.collection_exists(name):
                if self.read_only:
                    raise FileNotFoundError(
                        f"Qdrant collection is missing during read-only access: {name}"
                    )
                self.client.create_collection(
                    name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size, distance=self.distance
                    ),
                )
            else:
                info = self.client.get_collection(name)
                vectors = info.config.params.vectors
                if isinstance(vectors, dict):
                    raise ValueError(
                        f"Qdrant collection {name} uses named vectors; one unnamed vector is required"
                    )
                actual_size = int(vectors.size)
                actual_distance = vectors.distance
                if actual_size != self.vector_size or actual_distance != self.distance:
                    raise ValueError(
                        f"Qdrant collection schema mismatch for {name}: "
                        f"expected size={self.vector_size}, distance={self.distance}; "
                        f"got size={actual_size}, distance={actual_distance}"
                    )
            if self.mode != "server" or self.read_only:
                continue
            for field in (
                "dataset_name", "split", "sample_id", "session_id", "segment_id", "memory_tier",
                "model_id", "model_family", "campaign_id", "extraction_run_id", "batch_id", "assembly_id",
                "policy_version",
            ):
                self.client.create_payload_index(
                    name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    def _scroll(self, collection: str, conditions: Iterable[Any], *, with_vectors: bool) -> list[Any]:
        points: list[Any] = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                collection,
                scroll_filter=models.Filter(must=list(conditions)),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            points.extend(page)
            if offset is None:
                return points

    @staticmethod
    def _sample_conditions(dataset_name: str, split: str, sample_id: str) -> list[Any]:
        if not dataset_name or not split or not sample_id:
            raise ValueError("Qdrant operations require dataset_name, split, and sample_id")
        return [
            FactQdrantStore._match("dataset_name", dataset_name),
            FactQdrantStore._match("split", split),
            FactQdrantStore._match("sample_id", sample_id),
        ]

    @staticmethod
    def _match(key: str, value: str):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _vector(self, vector: Any) -> list[float]:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        if len(value) != self.vector_size:
            raise ValueError(f"vector size mismatch: expected {self.vector_size}, got {len(value)}")
        return [float(item) for item in value]

    def _require_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("Qdrant store was opened read-only")

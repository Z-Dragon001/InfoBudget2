"""Human-inspection exports that never participate in retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.ledger import atomic_write_json


def export_memories(
    store: FactQdrantStore,
    collection: str,
    *,
    dataset_name: str,
    split: str,
    sample_id: str,
    output_path: str | Path,
    assembly_id: str | None = None,
    extraction_run_id: str | None = None,
) -> Path:
    if collection == "assembled":
        points = store.assembly_points(dataset_name=dataset_name, split=split, sample_id=sample_id, assembly_id=assembly_id or "", with_vectors=False)
    else:
        if not extraction_run_id:
            raise ValueError("candidate export requires extraction_run_id")
        points = store.candidate_points(
            collection,
            dataset_name=dataset_name,
            split=split,
            sample_id=sample_id,
            extraction_run_id=extraction_run_id,
            with_vectors=False,
        )
    payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "split": split,
            "sample_id": sample_id,
            "collection_tier": collection,
            "assembly_id": assembly_id,
            "extraction_run_id": extraction_run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warning": "Human inspection only. Experiments must query Qdrant.",
        },
        "memories": [dict(point.payload or {}) for point in points],
    }
    path = Path(output_path)
    atomic_write_json(path, payload)
    return path

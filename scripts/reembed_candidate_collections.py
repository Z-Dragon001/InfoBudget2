"""Copy immutable candidate Facts into a new embedding-specific Qdrant namespace."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.embedding import LocalSentenceEncoder
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.manifest import memory_embedding_hash, resolve_collection_namespace
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.schemas import FactRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_jsonl", type=Path)
    parser.add_argument("--source-namespace", required=True)
    parser.add_argument("--source-vector-size", type=int, default=1024)
    parser.add_argument("--extraction-run-id", required=True)
    parser.add_argument("--tier", action="append", choices=["small", "medium", "large"])
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.source_vector_size <= 0:
        parser.error("--source-vector-size must be positive")

    bundle = load_rl_bundle(args.config_dir)
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    segments = load_segments(args.segments_jsonl)
    first = segments[0]
    embedding = bundle.embeddings["memory"]
    if int(embedding["dimension"]) != 384:
        raise ValueError("re-embedding target must be the configured 384-dimensional MiniLM")
    encoder = LocalSentenceEncoder(
        model_name=str(embedding["model_name"]),
        local_path=bundle.project.root_dir / str(embedding["local_path"]),
        dimension=int(embedding["dimension"]),
        normalize=bool(embedding["normalize"]),
        max_length=int(embedding.get("max_length", 256)),
        long_text_strategy=str(embedding.get("long_text_strategy", "truncate")),
    )
    target_embedding_hash = memory_embedding_hash(bundle)
    target_namespace = resolve_collection_namespace(
        bundle.rl["storage"],
        project_name=bundle.project.config.project.name,
        model_family=bundle.rl["model_family"],
        dataset=first.dataset_name,
        split=first.split,
        segmentation_version=first.segmentation_version,
        embedding_hash=target_embedding_hash,
    )
    if target_namespace == args.source_namespace:
        raise ValueError("source and target Qdrant namespaces must differ")

    source_config = dict(bundle.rl["storage"])
    source_config["vector_size"] = args.source_vector_size
    source = FactQdrantStore.from_storage_config(
        source_config,
        project_root=bundle.project.root_dir,
        namespace=args.source_namespace,
        read_only=True,
    )
    target = FactQdrantStore.from_storage_config(
        bundle.rl["storage"],
        project_root=bundle.project.root_dir,
        namespace=target_namespace,
    )
    selected_tiers = tuple(args.tier or ("small", "medium", "large"))
    counts = {}
    try:
        for tier in selected_tiers:
            points = source.candidate_points(
                tier,
                dataset_name=first.dataset_name,
                split=first.split,
                sample_id=first.sample_id,
                extraction_run_id=args.extraction_run_id,
                with_vectors=False,
            )
            facts = [FactRecord.from_payload(dict(point.payload or {})) for point in points]
            for fact in facts:
                fact.model_id = bundle.project.models[tier].stable_model_id
                fact.embedding_model = str(embedding["model_name"])
                fact.embedding_dimension = int(embedding["dimension"])
                fact.extra.update(
                    {
                        "embedding_model_hash": target_embedding_hash,
                        "embedding_revision": str(embedding.get("revision") or ""),
                        "embedding_normalized": bool(embedding["normalize"]),
                        "reembedded_from_namespace": args.source_namespace,
                        "reembedded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            vectors = encoder.encode([fact.fact_text for fact in facts]) if facts else []
            target.upsert_facts(tier, facts, vectors)
            stored = target.candidate_points(
                tier,
                dataset_name=first.dataset_name,
                split=first.split,
                sample_id=first.sample_id,
                extraction_run_id=args.extraction_run_id,
                with_vectors=False,
            )
            if len(stored) != len(facts):
                raise RuntimeError(
                    f"target Qdrant count mismatch for {tier}: expected={len(facts)}, actual={len(stored)}"
                )
            counts[tier] = len(facts)
    finally:
        source.close()
        target.close()
    manifest = {
        "schema_version": "candidate_reembedding_v1",
        "dataset_name": first.dataset_name,
        "split": first.split,
        "sample_id": first.sample_id,
        "extraction_run_id": args.extraction_run_id,
        "source_namespace": args.source_namespace,
        "source_vector_size": args.source_vector_size,
        "target_namespace": target_namespace,
        "target_embedding_model": embedding["model_name"],
        "target_embedding_revision": embedding.get("revision"),
        "target_embedding_dimension": embedding["dimension"],
        "target_embedding_hash": target_embedding_hash,
        "fact_counts": counts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.manifest:
        atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

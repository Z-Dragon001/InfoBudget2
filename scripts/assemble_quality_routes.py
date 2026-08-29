"""Assemble Qdrant S from capability-conditioned routing artifact D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.io import iter_jsonl
from infobudget.rl_router.assembly import AssemblyManager
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.export import export_memories
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.manifest import memory_embedding_hash, resolve_collection_namespace
from infobudget.rl_router.qdrant_store import FactQdrantStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_jsonl", type=Path)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--extraction-run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    bundle = load_rl_bundle(args.config_dir)
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    segments = load_segments(args.segments_jsonl)
    first = segments[0]
    decision_by_segment = {}
    for row in iter_jsonl(args.decisions):
        if (
            str(row.get("dataset") or row.get("dataset_name")) == first.dataset_name
            and str(row.get("split")) == first.split
            and str(row.get("sample_id")) == first.sample_id
        ):
            segment_id = str(row.get("segment_id") or "")
            if segment_id in decision_by_segment:
                raise ValueError(f"duplicate route decision for segment: {segment_id}")
            decision_by_segment[segment_id] = row
    expected_ids = {segment.segment_id for segment in segments}
    if set(decision_by_segment) != expected_ids:
        raise ValueError(
            "route decisions do not exactly cover the sample segments; "
            f"missing={sorted(expected_ids - set(decision_by_segment))}, "
            f"extra={sorted(set(decision_by_segment) - expected_ids)}"
        )
    decisions = [decision_by_segment[segment.segment_id] for segment in segments]
    actions = [str(row["selected_tier"]) for row in decisions]
    if any(action not in {"small", "medium", "large"} for action in actions):
        raise ValueError("selected_tier must be small, medium, or large")
    budget_run_ids = {str(row.get("budget_run_id") or "") for row in decisions}
    checkpoint_hashes = {str(row.get("quality_checkpoint_hash") or "") for row in decisions}
    if len(budget_run_ids) != 1 or "" in budget_run_ids:
        raise ValueError("sample decisions must share one non-empty budget_run_id")
    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("sample decisions must share one non-empty checkpoint hash")

    namespace = resolve_collection_namespace(
        bundle.rl["storage"],
        model_family=bundle.rl["model_family"],
        dataset=first.dataset_name,
        split=first.split,
        segmentation_version=first.segmentation_version,
        embedding_hash=memory_embedding_hash(bundle),
    )
    store = FactQdrantStore.from_storage_config(
        bundle.rl["storage"],
        project_root=bundle.project.root_dir,
        namespace=namespace,
    )
    output_dir = (
        args.output_dir
        or bundle.project.root_dir
        / "outputs"
        / "quality_router"
        / first.dataset_name
        / first.split
        / first.segmentation_method
        / "samples"
        / first.sample_id
    ).resolve()
    manager = AssemblyManager(store, output_dir / "routing" / "ledger.sqlite3")
    metadata_fields = (
        "selected_model_id",
        "selected_profile_id",
        "predicted_quality",
        "selected_cost",
        "route_decision_id",
        "quality_checkpoint_hash",
        "budget_run_id",
        "sample_budget",
        "sample_total_selected_cost",
    )
    result = manager.create(
        dataset_name=first.dataset_name,
        split=first.split,
        sample_id=first.sample_id,
        segments=segments,
        actions=actions,
        probabilities=None,
        episode_id=next(iter(budget_run_ids)),
        policy_version=next(iter(checkpoint_hashes))[:12],
        router_type="capability_conditioned_quality_v1",
        candidate_extraction_run_id=args.extraction_run_id,
        route_metadata=[
            {field: row[field] for field in metadata_fields if field in row}
            for row in decisions
        ],
    )
    if result.status != "ready":
        raise RuntimeError(f"assembly failed: {result}")
    export_memories(
        store,
        "assembled",
        dataset_name=first.dataset_name,
        split=first.split,
        sample_id=first.sample_id,
        assembly_id=result.assembly_id,
        output_path=output_dir / "human_readable" / "S_memories.json",
    )
    store.close()
    print(json.dumps({"assembly_id": result.assembly_id, "point_count": result.point_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Create a real S assembly for All-Small/Medium/Large or Random baselines."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from infobudget.rl_router.assembly import AssemblyManager
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.export import export_memories
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.manifest import memory_embedding_hash, resolve_collection_namespace
from infobudget.rl_router.qdrant_store import FactQdrantStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_jsonl", type=Path)
    parser.add_argument("--policy", choices=["all-small", "all-medium", "all-large", "random"], required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--extraction-run-id")
    args = parser.parse_args()
    bundle = load_rl_bundle(args.config_dir)
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    segments = load_segments(args.segments_jsonl)
    first = segments[0]
    storage = bundle.rl["storage"]
    namespace = resolve_collection_namespace(
        storage,
        project_name=bundle.project.config.project.name,
        model_family=bundle.rl["model_family"],
        dataset=first.dataset_name,
        split=first.split,
        segmentation_version=first.segmentation_version,
        embedding_hash=memory_embedding_hash(bundle),
    )
    store = FactQdrantStore.from_storage_config(
        storage,
        project_root=bundle.project.root_dir,
        namespace=namespace,
    )
    mapping = {"all-small": "small", "all-medium": "medium", "all-large": "large"}
    if args.policy == "random":
        rng = random.Random(bundle.rl["seed"])
        actions = [rng.choice(("small", "medium", "large")) for _ in segments]
    else:
        actions = [mapping[args.policy]] * len(segments)
    sample_root = bundle.project.root_dir / "outputs" / "rl_router" / first.dataset_name / first.split / first.segmentation_method / "samples" / first.sample_id
    manager = AssemblyManager(store, sample_root / "routing" / "ledger.sqlite3")
    result = manager.create(
        dataset_name=first.dataset_name,
        split=first.split,
        sample_id=first.sample_id,
        segments=segments,
        actions=actions,
        probabilities=[1.0 if args.policy != "random" else 1 / 3] * len(segments),
        episode_id=f"baseline-{args.policy}",
        policy_version=args.policy,
        router_type=args.policy,
        candidate_extraction_run_id=args.extraction_run_id,
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
        output_path=sample_root / "human_readable" / "S_memories.json",
    )
    store.close()
    print(result)


if __name__ == "__main__":
    main()

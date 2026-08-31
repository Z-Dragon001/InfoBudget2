"""Evaluate one physical S assembly with the exact LightMEM Reader/Judge prompts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from infobudget.datasets.loader import DatasetLoader
from infobudget.datasets.splits import load_split_selection
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.embedding import LocalSentenceEncoder
from infobudget.rl_router.evaluation import build_lightmem_evaluator
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.manifest import memory_embedding_hash, resolve_collection_namespace
from infobudget.rl_router.qdrant_store import FactQdrantStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_jsonl", type=Path)
    parser.add_argument("--assembly-id", required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--partition", choices=["train", "validation", "test"])
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()

    bundle = load_rl_bundle(args.config_dir)
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    segments = load_segments(args.segments_jsonl)
    first = segments[0]
    selection = None
    provided_split_args = (args.split_manifest, args.fold, args.partition)
    if any(value is not None for value in provided_split_args) and not all(
        value is not None for value in provided_split_args
    ):
        parser.error("--split-manifest, --fold, and --partition must be provided together")
    if args.split_manifest is None:
        parser.error("assembly evaluation requires a split manifest, fold, and partition")
    if args.split_manifest is not None:
        samples_dir = args.segments_jsonl.resolve().parents[1]
        available = {item.name for item in samples_dir.iterdir() if item.is_dir()}
        selection = load_split_selection(
            args.split_manifest,
            dataset_name=first.dataset_name,
            source_split=first.split,
            fold=args.fold,
            available_sample_ids=available,
            source_processed_manifest_path=(
                bundle.project.root_dir
                / "datasets"
                / "processed"
                / first.dataset_name
                / first.split
                / "manifest.json"
            ),
            project_root=bundle.project.root_dir,
        )
        if first.sample_id not in selection.sample_ids(args.partition):
            parser.error(
                f"sample {first.sample_id} is not in fold {selection.fold} partition {args.partition}"
            )
    examples = DatasetLoader(bundle.project.config.dataset, bundle.project.root_dir).load(
        first.dataset_name,
        first.split,
        {first.sample_id},
    )
    example = next((item for item in examples if item.sample_id == first.sample_id), None)
    if example is None:
        raise ValueError(f"processed sample is missing: {first.sample_id}")

    embedding = bundle.embeddings["memory"]
    encoder = LocalSentenceEncoder(
        model_name=embedding["model_name"],
        local_path=bundle.project.root_dir / embedding["local_path"],
        dimension=embedding["dimension"],
        normalize=embedding["normalize"],
        max_length=embedding.get("max_length"),
        long_text_strategy=embedding.get("long_text_strategy", "truncate"),
    )
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
    if selection is None:
        ledger_path = (
            bundle.project.root_dir
            / "outputs"
            / "rl_router"
            / first.dataset_name
            / first.split
            / first.segmentation_method
            / "samples"
            / first.sample_id
            / "qa"
            / "ledger.sqlite3"
        )
    else:
        ledger_path = (
            bundle.project.root_dir
            / "outputs"
            / "rl_router"
            / "evaluation"
            / first.dataset_name
            / selection.protocol
            / f"fold_{selection.fold}"
            / args.partition
            / first.segmentation_method
            / "samples"
            / first.sample_id
            / "ledger.sqlite3"
        )
    evaluator = build_lightmem_evaluator(
        bundle,
        store=store,
        encoder=encoder,
        ledger_path=ledger_path,
    )
    try:
        score, evaluations = evaluator.evaluate_sample(
            [pair.to_dict() for pair in example.qa_pairs],
            dataset_name=first.dataset_name,
            split=first.split,
            sample_id=first.sample_id,
            assembly_id=args.assembly_id,
            sample_metadata=example.metadata,
        )
    finally:
        store.close()
    print(
        json.dumps(
            {
                "dataset_name": first.dataset_name,
                "sample_id": first.sample_id,
                "assembly_id": args.assembly_id,
                "experiment_partition": args.partition,
                "split_manifest_sha256": selection.sha256 if selection is not None else None,
                "qa_score": score,
                "evaluations": [asdict(item) for item in evaluations],
                "ledger_path": str(ledger_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

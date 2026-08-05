"""Train the budget-constrained Embedding+MLP router over frozen L/M/H candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from infobudget.datasets.loader import DatasetLoader
from infobudget.datasets.splits import load_split_selection
from infobudget.rl_router.assembly import AssemblyManager
from infobudget.rl_router.campaign import load_complete_campaign
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.costs import normalize_virtual_cost, replay_virtual_cost
from infobudget.rl_router.embedding import LocalSentenceEncoder, LocalTokenizer
from infobudget.rl_router.evaluation import build_lightmem_evaluator
from infobudget.rl_router.experiment import RLExperimentTrainer
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.ledger import SqliteLedger
from infobudget.rl_router.manifest import (
    create_experiment_manifest,
    memory_embedding_hash,
    resolve_collection_namespace,
)
from infobudget.rl_router.parsing import render_extraction_prompt
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.reconciliation import reconcile_extraction_run
from infobudget.rl_router.router import EmbeddingMLPRouter, SegmentFeatureBuilder
from infobudget.rl_router.training import ConstrainedActorCriticTrainer
from infobudget.rl_router.training_io import discover_segment_files, load_replay_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["locomo", "longmemeval"])
    parser.add_argument("split")
    parser.add_argument("--method", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--extraction-run-id", help="Only valid when exactly one sample is selected.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-sample", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-version")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.steps_per_sample <= 0 or args.early_stopping_patience <= 0:
        parser.error("--epochs, --steps-per-sample, and --early-stopping-patience must be positive")
    if (args.split_manifest is None) != (args.fold is None):
        parser.error("--split-manifest and --fold must be provided together")
    if args.split_manifest and args.sample_ids:
        parser.error("--sample-id cannot be combined with --split-manifest")
    if args.split_manifest is None:
        parser.error("router training requires --split-manifest and --fold")

    bundle = load_rl_bundle(args.config_dir)
    _require_training_api_keys(bundle)
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    campaign = load_complete_campaign(bundle, args.campaign_id)
    if (
        campaign["dataset_name"],
        campaign["split"],
        campaign["segmentation_method"],
    ) != (args.dataset, args.split, args.method):
        raise ValueError("training scope does not match the extraction campaign")
    selection = None
    selected_sample_ids = args.sample_ids
    if args.split_manifest:
        available = set(
            discover_segment_files(
                bundle.project.root_dir,
                args.dataset,
                args.split,
                args.method,
            )
        )
        selection = load_split_selection(
            args.split_manifest,
            dataset_name=args.dataset,
            source_split=args.split,
            fold=args.fold,
            available_sample_ids=available,
            source_processed_manifest_path=(
                bundle.project.root_dir
                / "datasets"
                / "processed"
                / args.dataset
                / args.split
                / "manifest.json"
            ),
            project_root=bundle.project.root_dir,
        )
        selected_sample_ids = list(selection.train_sample_ids)
    segment_files = discover_segment_files(
        bundle.project.root_dir,
        args.dataset,
        args.split,
        args.method,
        selected_sample_ids,
    )
    validation_segment_files = (
        discover_segment_files(
            bundle.project.root_dir,
            args.dataset,
            args.split,
            args.method,
            list(selection.validation_sample_ids),
        )
        if selection is not None and selection.validation_sample_ids
        else {}
    )
    if args.extraction_run_id and len(segment_files) != 1:
        parser.error("--extraction-run-id requires exactly one selected sample")
    segments_by_sample = {sample_id: load_segments(path) for sample_id, path in segment_files.items()}
    validation_segments_by_sample = {
        sample_id: load_segments(path) for sample_id, path in validation_segment_files.items()
    }
    all_selected_segments = {**segments_by_sample, **validation_segments_by_sample}
    _validate_segment_sets(all_selected_segments, args.dataset, args.split, args.method)

    examples = {
        example.sample_id: example
        for example in DatasetLoader(bundle.project.config.dataset, bundle.project.root_dir).load(
            args.dataset,
            args.split,
            set(all_selected_segments),
        )
    }
    missing_examples = sorted(set(all_selected_segments) - examples.keys())
    if missing_examples:
        raise ValueError(f"processed samples are missing: {missing_examples}")
    if any(not examples[sample_id].qa_pairs for sample_id in all_selected_segments):
        raise ValueError("every selected train/validation sample must contain at least one QA pair")

    sample_root = bundle.project.root_dir / "outputs" / "rl_router" / args.dataset / args.split / args.method / "samples"
    histories = {}
    extraction_runs = {}
    for sample_id, segments in all_selected_segments.items():
        expected_run_id = campaign["expected_runs"].get(sample_id)
        if not expected_run_id:
            raise ValueError(f"sample {sample_id} is absent from extraction campaign")
        if args.extraction_run_id and args.extraction_run_id != expected_run_id:
            raise ValueError("--extraction-run-id does not match the extraction campaign")
        run_id, history = load_replay_history(
            sample_root / sample_id / "extraction" / "candidate_ledger.sqlite3",
            segments,
            expected_run_id,
        )
        extraction_runs[sample_id] = run_id
        histories[sample_id] = history

    reconciliation_first = next(iter(all_selected_segments.values()))[0]
    reconciliation_embedding_hash = memory_embedding_hash(bundle)
    reconciliation_namespace = resolve_collection_namespace(
        bundle.rl["storage"],
        dataset=args.dataset,
        split=args.split,
        segmentation_version=reconciliation_first.segmentation_version,
        embedding_hash=reconciliation_embedding_hash,
    )
    reconciliation_store = FactQdrantStore.from_storage_config(
        bundle.rl["storage"],
        project_root=bundle.project.root_dir,
        namespace=reconciliation_namespace,
        read_only=True,
    )
    reconciliation_results = {}
    try:
        for expected_run_id in sorted(set(extraction_runs.values())):
            reconciliation_results[expected_run_id] = reconcile_extraction_run(
                bundle.project.root_dir,
                expected_run_id,
                reconciliation_store,
                raise_on_error=True,
            )
    finally:
        reconciliation_store.close()

    run_id = datetime.now(timezone.utc).strftime("train_%Y%m%dT%H%M%SZ")
    if selection is None:
        default_output = bundle.project.root_dir / "outputs" / "rl_router" / "training" / args.dataset / args.split / args.method / run_id
    else:
        default_output = bundle.project.root_dir / "outputs" / "rl_router" / "training" / args.dataset / selection.protocol / f"fold_{selection.fold}" / args.method / run_id
    output_dir = (args.output_dir or default_output).resolve()
    if (output_dir / "training_ledger.sqlite3").exists() or (output_dir / "checkpoints").exists():
        raise FileExistsError(f"training output already exists; choose a new --output-dir: {output_dir}")

    router_encoder = _build_encoder(bundle, "router")
    memory_encoder = router_encoder if bundle.embeddings["router"] == bundle.embeddings["memory"] else _build_encoder(bundle, "memory")
    all_segments = [segment for sample_id in segments_by_sample for segment in segments_by_sample[sample_id]]
    feature_builder = SegmentFeatureBuilder(router_encoder)
    scaler = feature_builder.fit(all_segments)
    features_by_sample = {
        sample_id: feature_builder.build(segments)
        for sample_id, segments in segments_by_sample.items()
    }
    validation_features_by_sample = {
        sample_id: feature_builder.build(segments)
        for sample_id, segments in validation_segments_by_sample.items()
    }

    router_cfg = bundle.rl["router"]
    seed = int(bundle.rl["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = EmbeddingMLPRouter(
        router_encoder.dimension + SegmentFeatureBuilder.numeric_dimension,
        [int(value) for value in router_cfg["hidden_dimensions"]],
        float(router_cfg["dropout"]),
    )
    device = _resolve_device(args.device)
    model.to(device)
    optimizer = ConstrainedActorCriticTrainer(
        model,
        budget=float(router_cfg["budget"]),
        learning_rate=float(router_cfg["learning_rate"]),
        lambda_learning_rate=float(router_cfg["lambda_learning_rate"]),
        seed=seed,
    )

    first = all_segments[0]
    storage = bundle.rl["storage"]
    embedding_hash = memory_embedding_hash(bundle)
    namespace = resolve_collection_namespace(
        storage,
        dataset=args.dataset,
        split=args.split,
        segmentation_version=first.segmentation_version,
        embedding_hash=embedding_hash,
    )
    store = FactQdrantStore.from_storage_config(
        storage,
        project_root=bundle.project.root_dir,
        namespace=namespace,
    )
    store.freeze_candidates()
    assembly_manager = AssemblyManager(store, output_dir / "routing" / "ledger.sqlite3")
    evaluator = build_lightmem_evaluator(
        bundle,
        store=store,
        encoder=memory_encoder,
        ledger_path=output_dir / "qa" / "ledger.sqlite3",
    )
    validation_evaluator = (
        build_lightmem_evaluator(
            bundle,
            store=store,
            encoder=memory_encoder,
            ledger_path=output_dir / "validation" / "evaluations.sqlite3",
        )
        if validation_segments_by_sample
        else None
    )
    experiment = RLExperimentTrainer(
        model=model,
        scaler=scaler,
        trainer=optimizer,
        assembly_manager=assembly_manager,
        evaluator=evaluator,
        virtual_cost=None,
        output_dir=output_dir,
    )
    prices = {
        tier: bundle.project.prices[bundle.project.models[tier].model_name]
        for tier in ("small", "medium", "large")
    }
    prompt = bundle.fact_extraction_prompt_path(args.dataset).read_text(encoding="utf-8")
    tokenizers = {
        tier: LocalTokenizer(bundle.project.root_dir / bundle.project.models[tier].tokenizer_local_path)
        for tier in ("small", "medium", "large")
    }
    shared_prompt_tokens = {
        tier: tokenizers[tier].count(render_extraction_prompt(prompt, tier, []))
        for tier in tokenizers
    }
    virtual_costs = {
        sample_id: partial(
            replay_virtual_cost,
            segments,
            historical=histories[sample_id],
            buffer_config=bundle.rl["extraction"]["buffers"],
            prices=prices,
            shared_prompt_tokens=shared_prompt_tokens,
        )
        for sample_id, segments in all_selected_segments.items()
    }
    baselines = {}
    normalizers = {}
    for sample_id, segments in all_selected_segments.items():
        all_small = virtual_costs[sample_id](["small"] * len(segments)).total_cost
        all_large = virtual_costs[sample_id](["large"] * len(segments)).total_cost
        # Validate the normalization before any paid QA calls are made.
        normalize_virtual_cost(all_small, all_small, all_large)
        baselines[sample_id] = {"all_small": all_small, "all_large": all_large}
        normalizers[sample_id] = partial(normalize_virtual_cost, all_small=all_small, all_large=all_large)

    policy_version = args.policy_version or run_id
    create_experiment_manifest(
        bundle,
        output_dir / "manifest.json",
        experiment_id=run_id,
        dataset_name=args.dataset,
        split=args.split,
        segmentation_method=args.method,
        sample_ids=list(segments_by_sample),
        validation_sample_ids=list(validation_segments_by_sample),
        extraction_runs=extraction_runs,
        extraction_reconciliation=reconciliation_results,
        extraction_campaign_id=args.campaign_id,
        extraction_campaign_scope_hash=campaign["campaign_scope_hash"],
        epochs=args.epochs,
        steps_per_sample=args.steps_per_sample,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        cost_baselines=baselines,
        split_manifest=(
            {
                "path": str(selection.path),
                "sha256": selection.sha256,
                "protocol": selection.protocol,
                "fold": selection.fold,
                "train_sample_ids": list(selection.train_sample_ids),
                "validation_sample_ids": list(selection.validation_sample_ids),
                "test_sample_ids": list(selection.test_sample_ids),
            }
            if selection is not None
            else None
        ),
        precomputed_embedding_hash=embedding_hash,
    )
    planned_episodes = args.epochs * args.steps_per_sample * len(segments_by_sample)
    planned_questions = args.epochs * args.steps_per_sample * sum(
        len(examples[sample_id].qa_pairs) for sample_id in segments_by_sample
    )
    planned_validation_questions = args.epochs * sum(
        len(examples[sample_id].qa_pairs) for sample_id in validation_segments_by_sample
    )
    print(
        json.dumps(
            {
                "status": "starting",
                "samples": len(segments_by_sample),
                "segments": len(all_segments),
                "episodes": planned_episodes,
                "qa_reader_calls": planned_questions,
                "judge_calls": planned_questions,
                "validation_qa_reader_calls": planned_validation_questions,
                "validation_judge_calls": planned_validation_questions,
                "device": device,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    validation_ledger = SqliteLedger(
        output_dir / "validation" / "ledger.sqlite3",
        "epochs",
        ("epoch", "sample_id"),
        legacy_jsonl_path=output_dir / "validation" / "epochs.jsonl",
    )
    best_validation: tuple[float, float] | None = None
    epochs_without_improvement = 0
    completed_epochs = 0
    stopped_early = False
    try:
        for epoch in range(args.epochs):
            for sample_id, segments in segments_by_sample.items():
                history = experiment.train_sample(
                    features=features_by_sample[sample_id],
                    segments=segments,
                    questions=[pair.to_dict() for pair in examples[sample_id].qa_pairs],
                    steps=args.steps_per_sample,
                    policy_version=policy_version,
                    sample_metadata=examples[sample_id].metadata,
                    virtual_cost=virtual_costs[sample_id],
                    cost_normalizer=normalizers[sample_id],
                    candidate_extraction_run_id=extraction_runs[sample_id],
                    save_final=False,
                    track_best=not validation_segments_by_sample,
                )
                latest = history[-1]
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "sample_id": sample_id,
                            "qa_score": latest.qa_score,
                            "normalized_cost": latest.virtual_cost,
                            "reward": latest.reward,
                            "lambda": latest.lagrange_multiplier,
                        },
                        ensure_ascii=False,
                    )
                )
            completed_epochs = epoch + 1
            if validation_segments_by_sample:
                validation_score, validation_cost = _evaluate_validation_epoch(
                    epoch=completed_epochs,
                    model=model,
                    features_by_sample=validation_features_by_sample,
                    segments_by_sample=validation_segments_by_sample,
                    examples=examples,
                    assembly_manager=assembly_manager,
                    evaluator=validation_evaluator,
                    virtual_costs=virtual_costs,
                    normalizers=normalizers,
                    extraction_runs=extraction_runs,
                    policy_version=policy_version,
                    ledger=validation_ledger,
                )
                within_budget = validation_cost <= float(router_cfg["budget"])
                metric = (validation_score, -validation_cost)
                improved = within_budget and (best_validation is None or metric > best_validation)
                if improved:
                    best_validation = metric
                    epochs_without_improvement = 0
                    model.save_checkpoint(
                        output_dir / "checkpoints" / "best.pt",
                        scaler,
                        {
                            "selected_by": "validation",
                            "epoch": completed_epochs,
                            "validation_qa_score": validation_score,
                            "validation_normalized_cost": validation_cost,
                        },
                    )
                else:
                    epochs_without_improvement += 1
                print(
                    json.dumps(
                        {
                            "epoch": completed_epochs,
                            "partition": "validation",
                            "qa_score": validation_score,
                            "normalized_cost": validation_cost,
                            "within_budget": within_budget,
                            "best": improved,
                        },
                        ensure_ascii=False,
                    )
                )
                if epochs_without_improvement >= args.early_stopping_patience:
                    stopped_early = True
                    break
        checkpoint = experiment.save_final(
            {
                "dataset_name": args.dataset,
                "split": args.split,
                "segmentation_method": args.method,
                "epochs_requested": args.epochs,
                "epochs_completed": completed_epochs,
                "steps_per_sample": args.steps_per_sample,
                "policy_version": policy_version,
                "stopped_early": stopped_early,
                "best_validation_qa_score": best_validation[0] if best_validation else None,
                "best_validation_normalized_cost": -best_validation[1] if best_validation else None,
            }
        )
    finally:
        store.close()
    print(json.dumps({"status": "complete", "checkpoint": str(checkpoint), "output_dir": str(output_dir)}, ensure_ascii=False))


def _evaluate_validation_epoch(
    *,
    epoch: int,
    model,
    features_by_sample,
    segments_by_sample,
    examples,
    assembly_manager,
    evaluator,
    virtual_costs,
    normalizers,
    extraction_runs,
    policy_version: str,
    ledger: SqliteLedger,
) -> tuple[float, float]:
    total_correct = 0.0
    total_questions = 0
    normalized_costs = []
    for sample_id, segments in segments_by_sample.items():
        routes = model.route(features_by_sample[sample_id], deterministic=True)
        actions = [route.tier for route in routes]
        assembly = assembly_manager.create(
            dataset_name=segments[0].dataset_name,
            split=segments[0].split,
            sample_id=sample_id,
            segments=segments,
            actions=actions,
            probabilities=[route.probability for route in routes],
            episode_id=f"validation:{epoch:04d}:{sample_id}",
            policy_version=policy_version,
            router_type="embedding_mlp",
            candidate_extraction_run_id=extraction_runs[sample_id],
        )
        if assembly.status != "ready":
            raise RuntimeError(f"validation assembly failed for sample {sample_id}")
        try:
            questions = [pair.to_dict() for pair in examples[sample_id].qa_pairs]
            qa_score, _ = evaluator.evaluate_sample(
                questions,
                dataset_name=segments[0].dataset_name,
                split=segments[0].split,
                sample_id=sample_id,
                assembly_id=assembly.assembly_id,
                sample_metadata=examples[sample_id].metadata,
            )
            normalized_cost = normalizers[sample_id](virtual_costs[sample_id](actions).total_cost)
            total_correct += qa_score * len(questions)
            total_questions += len(questions)
            normalized_costs.append(normalized_cost)
            ledger.append(
                {
                    "epoch": epoch,
                    "sample_id": sample_id,
                    "assembly_id": assembly.assembly_id,
                    "qa_score": qa_score,
                    "normalized_cost": normalized_cost,
                    "num_questions": len(questions),
                    "route_decisions": actions,
                    "candidate_extraction_run_id": extraction_runs[sample_id],
                }
            )
        finally:
            assembly_manager.cleanup(
                assembly,
                dataset_name=segments[0].dataset_name,
                split=segments[0].split,
            )
    return total_correct / total_questions, sum(normalized_costs) / len(normalized_costs)


def _build_encoder(bundle, role: str) -> LocalSentenceEncoder:
    config = bundle.embeddings[role]
    return LocalSentenceEncoder(
        model_name=config["model_name"],
        local_path=bundle.project.root_dir / config["local_path"],
        dimension=config["dimension"],
        normalize=config["normalize"],
    )


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return value


def _require_training_api_keys(bundle) -> None:
    missing = [
        bundle.project.models[role].api_key_env
        for role in ("qa_reader", "judge_llm")
        if not bundle.project.models[role].resolved_api_key()
    ]
    if missing:
        raise RuntimeError(f"missing training API key environment variables: {missing}")


def _validate_segment_sets(segments_by_sample, dataset_name: str, split: str, method: str) -> None:
    versions = set()
    for sample_id, segments in segments_by_sample.items():
        for segment in segments:
            if (segment.dataset_name, segment.split, segment.sample_id, segment.segmentation_method) != (
                dataset_name,
                split,
                sample_id,
                method,
            ):
                raise ValueError(f"segment scope mismatch in sample {sample_id}")
            versions.add(segment.segmentation_version)
    if len(versions) != 1:
        raise ValueError(f"training samples mix segmentation versions: {sorted(versions)}")


if __name__ == "__main__":
    main()

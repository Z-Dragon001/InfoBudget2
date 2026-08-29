"""Predict scalar local quality and emit budget-feasible routing artifact D."""

from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from infobudget.quality_router.budget import optimize_budget
from infobudget.quality_router.config import QualityRouterConfig
from infobudget.quality_router.io import (
    file_sha256,
    iter_jsonl,
    load_capability_profiles,
    write_jsonl,
)
from infobudget.quality_router.model import (
    CapabilityConditionedQualityScorer,
    QualityFeatureBuilder,
)
from infobudget.quality_router.schemas import FactSetKey, QualityPrediction
from infobudget.rl_router.embedding import LocalSentenceEncoder
from infobudget.rl_router.schemas import TopicSegment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=float, required=True, help="Absolute budget applied independently to each sample.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--budget-run-id")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget cannot be negative")

    quality_config = QualityRouterConfig.load(args.config_dir / "quality_router.yaml")
    embedding = _embedding_config(args.config_dir / "embeddings.yaml")
    project_root = args.config_dir.resolve().parent
    encoder = LocalSentenceEncoder(
        model_name=str(embedding["model_name"]),
        local_path=project_root / str(embedding["local_path"]),
        dimension=int(embedding["dimension"]),
        normalize=bool(embedding.get("normalize", True)),
        max_length=int(embedding.get("max_length", 256)),
        long_text_strategy=str(embedding.get("long_text_strategy", "mean_pool_chunks")),
    )
    device = _resolve_device(args.device)
    model, scaler, checkpoint_metadata = CapabilityConditionedQualityScorer.load_checkpoint(
        args.checkpoint, device=device
    )
    if checkpoint_metadata["embedding_model"] != embedding["model_name"] or checkpoint_metadata["embedding_dimension"] != int(embedding["dimension"]):
        raise ValueError("checkpoint and configured embedding model do not match")
    profiles = load_capability_profiles(args.capabilities)
    segments = _load_segments(args.segments)
    costs, tiers = _load_costs(args.costs)
    missing_segment_costs = sorted(set(costs) - segments.keys())
    if missing_segment_costs:
        raise ValueError(f"cost rows reference missing segments: {missing_segment_costs[:10]}")

    feature_builder = QualityFeatureBuilder(encoder, scaler)
    keys = sorted(costs)
    segment_list = [segments[key] for key in keys]
    segment_features = feature_builder.build_segment_features(segment_list)
    predictions: list[QualityPrediction] = []
    prediction_rows: list[dict] = []
    for key, segment_feature in zip(keys, segment_features):
        for model_id, cost in sorted(costs[key].items()):
            if model_id not in profiles:
                raise ValueError(f"capability profile is missing for cost model: {model_id}")
            pair_features = feature_builder.combine(
                np.asarray([segment_feature], dtype=np.float32), [profiles[model_id]]
            )
            predicted_quality = float(model.predict(pair_features, device=device)[0])
            prediction = QualityPrediction(
                key=FactSetKey(*key),
                model_id=model_id,
                profile_id=profiles[model_id].profile_id,
                predicted_quality=predicted_quality,
                cost=cost,
            )
            predictions.append(prediction)
            prediction_rows.append(
                {
                    "dataset": key[0],
                    "split": key[1],
                    "sample_id": key[2],
                    "segment_id": key[3],
                    "model_id": model_id,
                    "profile_id": prediction.profile_id,
                    "tier": tiers[(key, model_id)],
                    "predicted_quality": predicted_quality,
                    "cost": cost,
                }
            )

    by_sample: dict[tuple[str, str, str], list[QualityPrediction]] = defaultdict(list)
    for prediction in predictions:
        by_sample[(prediction.key.dataset, prediction.key.split, prediction.key.sample_id)].append(prediction)
    budget_run_id = args.budget_run_id or f"quality_budget_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    checkpoint_hash = file_sha256(args.checkpoint)
    output_rows = []
    for sample_key, sample_predictions in sorted(by_sample.items()):
        solution = optimize_budget(
            sample_predictions,
            budget=args.budget,
            cost_quantum=float(quality_config.values["budget_cost_quantum"]),
        )
        all_by_segment: dict[str, list[QualityPrediction]] = defaultdict(list)
        for prediction in sample_predictions:
            all_by_segment[prediction.key.segment_id].append(prediction)
        for selected in solution.selections:
            key = selected.key.tuple()
            output_rows.append(
                {
                    "schema_version": "routing_decision_v1",
                    "dataset": key[0],
                    "split": key[1],
                    "sample_id": key[2],
                    "segment_id": key[3],
                    "candidate_predictions": [
                        {
                            "model_id": item.model_id,
                            "profile_id": item.profile_id,
                            "tier": tiers[(key, item.model_id)],
                            "predicted_quality": item.predicted_quality,
                            "cost": item.cost,
                        }
                        for item in sorted(all_by_segment[key[3]], key=lambda value: value.model_id)
                    ],
                    "selected_model_id": selected.model_id,
                    "selected_profile_id": selected.profile_id,
                    "selected_tier": tiers[(key, selected.model_id)],
                    "predicted_quality": selected.predicted_quality,
                    "selected_cost": selected.cost,
                    "sample_budget": args.budget,
                    "sample_total_selected_cost": solution.total_cost,
                    "sample_total_predicted_quality": solution.total_quality,
                    "route_decision_id": f"{budget_run_id}:{key[2]}:{key[3]}",
                    "budget_run_id": budget_run_id,
                    "quality_checkpoint_hash": checkpoint_hash,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    write_jsonl(args.output, output_rows)
    if args.predictions_output:
        write_jsonl(args.predictions_output, prediction_rows)
    print(json.dumps({"decisions": len(output_rows), "budget_run_id": budget_run_id, "output": str(args.output.resolve())}, ensure_ascii=False))


def _embedding_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = payload.get("embeddings", {}).get("router")
    if not isinstance(value, dict):
        raise ValueError("embeddings.router is missing")
    if value.get("model_name") != "sentence-transformers/all-MiniLM-L6-v2" or int(value.get("dimension", 0)) != 384:
        raise ValueError("routing requires all-MiniLM-L6-v2 with 384 dimensions")
    return value


def _load_segments(path: Path) -> dict[tuple[str, str, str, str], TopicSegment]:
    result = {}
    for row in iter_jsonl(path):
        if not {"dataset_name", "split", "sample_id", "segment_id", "text", "turn_ids"}.issubset(row):
            continue
        segment = TopicSegment.from_dict(row)
        key = (segment.dataset_name, segment.split, segment.sample_id, segment.segment_id)
        if key in result:
            raise ValueError(f"duplicate segment: {key}")
        result[key] = segment
    return result


def _load_costs(path: Path) -> tuple[dict, dict]:
    costs: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    tiers = {}
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row).tuple()
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        if not model_id:
            raise ValueError("cost row is missing model_id/extractor_model")
        cost = float(row.get("cost", row.get("allocated_total_cost", -1)))
        if cost < 0:
            raise ValueError("cost row is missing a non-negative cost")
        if model_id in costs[key]:
            raise ValueError(f"duplicate segment/model cost: {key}, {model_id}")
        costs[key][model_id] = cost
        tier = str(row.get("tier") or row.get("memory_tier") or "").strip()
        if tier not in {"small", "medium", "large"}:
            raise ValueError("cost row tier must be small, medium, or large")
        tiers[(key, model_id)] = tier
    if not costs:
        raise ValueError("no cost rows found")
    return dict(costs), tiers


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return value


if __name__ == "__main__":
    main()

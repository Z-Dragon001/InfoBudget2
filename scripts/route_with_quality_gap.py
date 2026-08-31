"""Predict local Fact quality and route with epsilon noninferiority."""

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

from infobudget.quality_gap_router.artifacts import group_predictions, load_costs
from infobudget.quality_gap_router.decision import (
    QualityGapPolicy,
    select_quality_gap_model,
)
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
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ood-flags", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--gap-run-id")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("schema_version") != "quality_gap_calibration_v1":
        raise ValueError("calibration schema_version must be quality_gap_calibration_v1")
    if not calibration.get("constraints_satisfied", False):
        raise ValueError("calibration did not satisfy its declared validation constraints")
    policy = QualityGapPolicy.from_dict(calibration)
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
    if (
        checkpoint_metadata["embedding_model"] != embedding["model_name"]
        or checkpoint_metadata["embedding_dimension"] != int(embedding["dimension"])
    ):
        raise ValueError("checkpoint and configured embedding model do not match")
    profiles = load_capability_profiles(args.capabilities)
    segments = _load_segments(args.segments)
    costs, tiers = load_costs(args.costs)
    missing_tiers = sorted(set(costs) - set(tiers))
    if missing_tiers:
        raise ValueError(f"routing cost rows are missing tiers: {missing_tiers[:10]}")
    unknown_cost_segments = sorted({identity[0] for identity in costs} - set(segments))
    if unknown_cost_segments:
        raise ValueError(f"cost rows reference missing segments: {unknown_cost_segments[:10]}")

    feature_builder = QualityFeatureBuilder(encoder, scaler)
    predictions = _predict_candidates(
        segments=segments,
        costs=costs,
        profiles=profiles,
        feature_builder=feature_builder,
        model=model,
        device=device,
    )
    grouped = group_predictions(predictions)
    expected_keys = {FactSetKey(*key) for key in segments}
    if set(grouped) != expected_keys:
        raise ValueError("predictions do not exactly cover every routed segment")
    ood_keys = _load_ood_flags(args.ood_flags) if args.ood_flags else set()
    unknown_ood = sorted(key.tuple() for key in ood_keys - expected_keys)
    if unknown_ood:
        raise ValueError(f"OOD flags reference missing segments: {unknown_ood[:10]}")

    decisions = {
        key: select_quality_gap_model(
            rows,
            policy=policy,
            segment_ood=key in ood_keys,
        )
        for key, rows in grouped.items()
    }
    selected_cost_by_sample: dict[tuple[str, str, str], float] = defaultdict(float)
    for decision in decisions.values():
        sample_key = (
            decision.key.dataset,
            decision.key.split,
            decision.key.sample_id,
        )
        selected_cost_by_sample[sample_key] += decision.selected.cost

    gap_run_id = args.gap_run_id or (
        f"quality_gap_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    checkpoint_hash = file_sha256(args.checkpoint)
    calibration_hash = file_sha256(args.calibration)
    output_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for key in sorted(grouped, key=lambda item: item.tuple()):
        decision = decisions[key]
        identity = key.tuple()
        sample_key = identity[:3]
        candidates = []
        for prediction in grouped[key]:
            candidate_identity = (identity, prediction.model_id)
            candidate_row = {
                "model_id": prediction.model_id,
                "profile_id": prediction.profile_id,
                "tier": tiers[candidate_identity],
                "predicted_quality": prediction.predicted_quality,
                "cost": prediction.cost,
                "predicted_gap": decision.predicted_gaps[prediction.model_id],
                "gap_upper_bound": decision.gap_upper_bounds[prediction.model_id],
                "eligible": prediction.model_id in decision.eligible_model_ids,
            }
            candidates.append(candidate_row)
            prediction_rows.append(
                {
                    "dataset": identity[0],
                    "split": identity[1],
                    "sample_id": identity[2],
                    "segment_id": identity[3],
                    **candidate_row,
                }
            )
        selected_identity = (identity, decision.selected.model_id)
        output_rows.append(
            {
                "schema_version": "quality_gap_routing_decision_v1",
                **decision.to_dict(),
                "selected_tier": tiers[selected_identity],
                "candidate_predictions": candidates,
                "epsilon": policy.epsilon,
                "quality_floor": policy.quality_floor,
                "uncertainty_enabled": policy.uncertainty_enabled,
                "gap_residual_bound": policy.gap_residual_bound,
                "sample_total_selected_cost": selected_cost_by_sample[sample_key],
                "route_decision_id": f"{gap_run_id}:{identity[2]}:{identity[3]}",
                "gap_run_id": gap_run_id,
                # Kept for compatibility with assemble_quality_routes.py.
                "budget_run_id": gap_run_id,
                "quality_checkpoint_hash": checkpoint_hash,
                "quality_gap_calibration_hash": calibration_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    write_jsonl(args.output, output_rows)
    if args.predictions_output:
        write_jsonl(args.predictions_output, prediction_rows)
    print(
        json.dumps(
            {
                "decisions": len(output_rows),
                "gap_run_id": gap_run_id,
                "epsilon": policy.epsilon,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


def _predict_candidates(
    *,
    segments: dict[tuple[str, str, str, str], TopicSegment],
    costs: dict[tuple, float],
    profiles: dict,
    feature_builder: QualityFeatureBuilder,
    model: CapabilityConditionedQualityScorer,
    device: str,
) -> list[QualityPrediction]:
    segment_keys = sorted({identity[0] for identity in costs})
    segment_list = [segments[key] for key in segment_keys]
    segment_features = feature_builder.build_segment_features(segment_list)
    feature_by_key = dict(zip(segment_keys, segment_features))
    predictions: list[QualityPrediction] = []
    for identity in sorted(costs):
        key_tuple, model_id = identity
        if model_id not in profiles:
            raise ValueError(f"capability profile is missing for cost model: {model_id}")
        pair_features = feature_builder.combine(
            np.asarray([feature_by_key[key_tuple]], dtype=np.float32),
            [profiles[model_id]],
        )
        quality = float(model.predict(pair_features, device=device)[0])
        predictions.append(
            QualityPrediction(
                key=FactSetKey(*key_tuple),
                model_id=model_id,
                profile_id=profiles[model_id].profile_id,
                predicted_quality=quality,
                cost=costs[identity],
            )
        )
    return predictions


def _embedding_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = payload.get("embeddings", {}).get("router")
    if not isinstance(value, dict):
        raise ValueError("embeddings.router is missing")
    if (
        value.get("model_name") != "sentence-transformers/all-MiniLM-L6-v2"
        or int(value.get("dimension", 0)) != 384
    ):
        raise ValueError("quality-gap routing requires all-MiniLM-L6-v2 with 384 dimensions")
    return value


def _load_segments(path: Path) -> dict[tuple[str, str, str, str], TopicSegment]:
    result = {}
    for row in iter_jsonl(path):
        required = {"dataset_name", "split", "sample_id", "segment_id", "text", "turn_ids"}
        if not required.issubset(row):
            continue
        segment = TopicSegment.from_dict(row)
        key = (
            segment.dataset_name,
            segment.split,
            segment.sample_id,
            segment.segment_id,
        )
        if key in result:
            raise ValueError(f"duplicate segment: {key}")
        result[key] = segment
    if not result:
        raise ValueError("no topic segments found")
    return result


def _load_ood_flags(path: Path) -> set[FactSetKey]:
    result: set[FactSetKey] = set()
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row)
        if bool(row.get("segment_ood", row.get("is_ood", row.get("ood", False)))):
            if key in result:
                raise ValueError(f"duplicate OOD flag: {key.tuple()}")
            result.add(key)
    return result


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return value


if __name__ == "__main__":
    main()

"""Train the capability-conditioned scalar quality scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

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
from infobudget.quality_router.schemas import FactQualityLabel
from infobudget.quality_router.training import (
    QualityTrainingConfig,
    train_quality_scorer,
)
from infobudget.rl_router.embedding import LocalSentenceEncoder
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.schemas import TopicSegment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    quality_config = QualityRouterConfig.load(args.config_dir / "quality_router.yaml")
    embedding_config = _embedding_config(args.config_dir / "embeddings.yaml")
    if int(embedding_config["dimension"]) != 384:
        raise ValueError("quality router requires all-MiniLM-L6-v2 with 384 dimensions")
    encoder = _build_encoder(embedding_config, args.config_dir.parent)
    profiles = load_capability_profiles(args.capabilities)
    segments = _load_segments(args.segments)
    train_labels = _load_labels(args.train_labels)
    validation_labels = _load_labels(args.validation_labels)
    _validate_group_split(train_labels, validation_labels)

    train_segments = _segments_for_labels(train_labels, segments)
    validation_segments = _segments_for_labels(validation_labels, segments)
    feature_builder = QualityFeatureBuilder(encoder)
    scaler = feature_builder.fit(_unique_segments(train_segments))
    train_features = _pair_features(feature_builder, train_segments, train_labels, profiles)
    validation_features = _pair_features(
        feature_builder, validation_segments, validation_labels, profiles
    )
    train_targets = np.asarray(
        [label.silver_strict_fact_f1 for label in train_labels], dtype=np.float32
    )
    validation_targets = np.asarray(
        [label.silver_strict_fact_f1 for label in validation_labels], dtype=np.float32
    )

    values = quality_config.values
    seed = int(values["seed"])
    torch.manual_seed(seed)
    model = CapabilityConditionedQualityScorer(
        feature_builder.input_dimension,
        [int(item) for item in values["hidden_dimensions"]],
        float(values["dropout"]),
    )
    device = _resolve_device(args.device)
    result = train_quality_scorer(
        model,
        train_features=train_features,
        train_targets=train_targets,
        validation_features=validation_features,
        validation_targets=validation_targets,
        config=QualityTrainingConfig(
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            batch_size=int(values["batch_size"]),
            epochs=int(values["epochs"]),
            early_stopping_patience=int(values["early_stopping_patience"]),
            huber_delta=float(values["huber_delta"]),
            seed=seed,
        ),
        device=device,
    )

    output_dir = args.output_dir.resolve()
    checkpoint = output_dir / "quality_scorer.pt"
    model.save_checkpoint(
        checkpoint,
        scaler=scaler,
        embedding_model=str(embedding_config["model_name"]),
        embedding_dimension=int(embedding_config["dimension"]),
        metadata={
            "label_name": values["label_name"],
            "train_labels_sha256": file_sha256(args.train_labels),
            "validation_labels_sha256": file_sha256(args.validation_labels),
            "capabilities_sha256": file_sha256(args.capabilities),
            "best_epoch": result.best_epoch,
            "best_validation_mae": result.best_validation_mae,
        },
    )
    validation_predictions = model.predict(validation_features, device=device)
    write_jsonl(
        output_dir / "validation_predictions.jsonl",
        (
            {
                **label.to_dict(),
                "predicted_quality": float(prediction),
                "absolute_error": abs(float(prediction) - label.silver_strict_fact_f1),
            }
            for label, prediction in zip(validation_labels, validation_predictions)
        ),
    )
    atomic_write_json(
        output_dir / "training_metrics.json",
        {
            "schema_version": "quality_training_metrics_v1",
            "embedding_model": embedding_config["model_name"],
            "embedding_dimension": embedding_config["dimension"],
            "input_dimension": feature_builder.input_dimension,
            "train_rows": len(train_labels),
            "validation_rows": len(validation_labels),
            "best_epoch": result.best_epoch,
            "best_validation_mae": result.best_validation_mae,
            "history": list(result.history),
        },
    )
    print(json.dumps({"checkpoint": str(checkpoint), "validation_mae": result.best_validation_mae}, ensure_ascii=False))


def _embedding_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = payload.get("embeddings", {}).get("router")
    if not isinstance(config, dict):
        raise ValueError("embeddings.router is missing")
    if config.get("model_name") != "sentence-transformers/all-MiniLM-L6-v2":
        raise ValueError("quality router embedding model must be sentence-transformers/all-MiniLM-L6-v2")
    return config


def _build_encoder(config: dict, project_root: Path) -> LocalSentenceEncoder:
    local_path = Path(str(config["local_path"]))
    return LocalSentenceEncoder(
        model_name=str(config["model_name"]),
        local_path=local_path if local_path.is_absolute() else project_root / local_path,
        dimension=int(config["dimension"]),
        normalize=bool(config.get("normalize", True)),
        max_length=int(config.get("max_length", 256)),
        long_text_strategy=str(config.get("long_text_strategy", "mean_pool_chunks")),
    )


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
    if not result:
        raise ValueError("no topic segments found")
    return result


def _load_labels(path: Path) -> list[FactQualityLabel]:
    labels = [FactQualityLabel.from_dict(row) for row in iter_jsonl(path)]
    keys = [(label.key.tuple(), label.model_id) for label in labels]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate segment/model label in {path}")
    if not labels:
        raise ValueError(f"no quality labels found: {path}")
    return labels


def _validate_group_split(train: list[FactQualityLabel], validation: list[FactQualityLabel]) -> None:
    train_groups = {(item.key.dataset, item.key.sample_id) for item in train}
    validation_groups = {(item.key.dataset, item.key.sample_id) for item in validation}
    overlap = sorted(train_groups & validation_groups)
    if overlap:
        raise ValueError(f"sample-level train/validation leakage: {overlap[:10]}")


def _segments_for_labels(labels: list[FactQualityLabel], segments: dict) -> list[TopicSegment]:
    missing = sorted({label.key.tuple() for label in labels} - segments.keys())
    if missing:
        raise ValueError(f"segments are missing for labels: {missing[:10]}")
    return [segments[label.key.tuple()] for label in labels]


def _unique_segments(segments: list[TopicSegment]) -> list[TopicSegment]:
    result = {}
    for segment in segments:
        result.setdefault((segment.dataset_name, segment.split, segment.sample_id, segment.segment_id), segment)
    return list(result.values())


def _pair_features(builder, segments, labels, profiles) -> np.ndarray:
    unique = _unique_segments(segments)
    unique_features = builder.build_segment_features(unique)
    feature_by_key = {
        (segment.dataset_name, segment.split, segment.sample_id, segment.segment_id): feature
        for segment, feature in zip(unique, unique_features)
    }
    missing_models = sorted({label.model_id for label in labels} - profiles.keys())
    if missing_models:
        raise ValueError(f"capability profiles are missing models: {missing_models}")
    for label in labels:
        if label.profile_id != profiles[label.model_id].profile_id:
            raise ValueError(f"label/profile mismatch for model {label.model_id}")
    segment_rows = np.asarray([feature_by_key[label.key.tuple()] for label in labels], dtype=np.float32)
    return builder.combine(segment_rows, [profiles[label.model_id] for label in labels])


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return value


if __name__ == "__main__":
    main()

"""Orchestrate predeclared-fold router training and real routed test evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.deployment import summarize_question_outcomes
from infobudget.rl_router.experiment_identity import epoch_artifact_name
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.manifest import file_sha256
from infobudget.rl_router.metrics import summarize_fold_accuracy
from infobudget.utils.logging import get_logger


logger = get_logger("routed_cv_evaluation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["locomo", "longmemeval"])
    parser.add_argument("split")
    parser.add_argument("--method", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-sample", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.steps_per_sample <= 0 or args.early_stopping_patience <= 0:
        parser.error("--epochs, --steps-per-sample, and --early-stopping-patience must be positive")

    bundle = load_rl_bundle(args.config_dir)
    split_path = args.split_manifest.resolve()
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    segmentation_manifest_path = (
        bundle.project.root_dir
        / "datasets"
        / "segmented"
        / args.dataset
        / args.split
        / args.method
        / "manifest.json"
    )
    if not segmentation_manifest_path.is_file():
        raise FileNotFoundError(
            f"segmentation manifest is missing: {segmentation_manifest_path}"
        )
    segmentation_manifest = json.loads(
        segmentation_manifest_path.read_text(encoding="utf-8")
    )
    if (
        segmentation_manifest.get("dataset_name"),
        segmentation_manifest.get("split"),
        segmentation_manifest.get("segmentation_method"),
    ) != (args.dataset, args.split, args.method):
        raise ValueError("segmentation manifest scope does not match the experiment")
    segmentation_manifest_sha256 = file_sha256(segmentation_manifest_path)
    declared_folds = [int(item["fold"]) for item in split_payload.get("folds", [])]
    folds = args.folds or declared_folds
    if not folds or len(folds) != len(set(folds)):
        parser.error("selected folds must be non-empty and unique")
    unknown = sorted(set(folds) - set(declared_folds))
    if unknown:
        parser.error(f"split manifest does not contain folds: {unknown}")

    experiment_id = args.experiment_id or (
        datetime.now(timezone.utc).strftime(f"routed_cv_e{args.epochs}_%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    default_output = (
        bundle.project.root_dir
        / "outputs"
        / "rl_router"
        / "full_experiments"
        / args.dataset
        / str(split_payload.get("protocol") or "split")
        / args.method
        / epoch_artifact_name(args.epochs)
        / experiment_id
    )
    output_dir = (args.output_dir or default_output).resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"experiment already exists; add --resume: {output_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "dataset_name": args.dataset,
            "split": args.split,
            "segmentation_method": args.method,
            "segmentation_manifest_sha256": segmentation_manifest_sha256,
            "adaptive_alpha": (segmentation_manifest.get("parameters") or {}).get(
                "adaptive_alpha"
            ),
            "split_manifest_sha256": file_sha256(split_path),
            "campaign_id": args.campaign_id,
            "folds": folds,
            "epochs": args.epochs,
            "steps_per_sample": args.steps_per_sample,
            "early_stopping_patience": args.early_stopping_patience,
            "device": args.device,
        }
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"full-experiment resume scope mismatch: {actual}")
    else:
        manifest = {
            "schema_version": "routed_cv_experiment_v1",
            "experiment_id": experiment_id,
            "status": "planned",
            "dataset_name": args.dataset,
            "split": args.split,
            "segmentation_method": args.method,
            "segmentation_algorithm": segmentation_manifest.get(
                "segmentation_algorithm", args.method
            ),
            "segmentation_version": segmentation_manifest.get("segmentation_version"),
            "segmentation_manifest": str(segmentation_manifest_path.resolve()),
            "segmentation_manifest_sha256": segmentation_manifest_sha256,
            "adaptive_alpha": (segmentation_manifest.get("parameters") or {}).get(
                "adaptive_alpha"
            ),
            "protocol": split_payload.get("protocol"),
            "split_manifest": str(split_path),
            "split_manifest_sha256": file_sha256(split_path),
            "campaign_id": args.campaign_id,
            "folds": folds,
            "epochs": args.epochs,
            "steps_per_sample": args.steps_per_sample,
            "early_stopping_patience": args.early_stopping_patience,
            "device": args.device,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fold_results": {},
        }
        atomic_write_json(manifest_path, manifest)

    scripts_dir = Path(__file__).resolve().parent
    config_dir = args.config_dir.resolve()
    manifest["status"] = "active"
    atomic_write_json(manifest_path, manifest)
    try:
        for fold in folds:
            fold_root = output_dir / f"fold_{fold}"
            training_dir = fold_root / "training"
            deployment_dir = fold_root / "deployment"
            training_manifest = training_dir / "manifest.json"
            deployment_manifest = deployment_dir / "manifest.json"
            train_run_id = f"{experiment_id}_fold_{fold}_train"
            deployment_run_id = f"{experiment_id}_fold_{fold}_test"
            if not _is_complete(training_manifest):
                training_args = [
                    scripts_dir / "train_rl_router.py",
                    args.dataset,
                    args.split,
                    "--method",
                    args.method,
                    "--split-manifest",
                    split_path,
                    "--fold",
                    fold,
                    "--campaign-id",
                    args.campaign_id,
                    "--epochs",
                    args.epochs,
                    "--steps-per-sample",
                    args.steps_per_sample,
                    "--early-stopping-patience",
                    args.early_stopping_patience,
                    "--device",
                    args.device,
                    "--output-dir",
                    training_dir,
                    "--config-dir",
                    config_dir,
                ]
                if training_manifest.exists():
                    training_args.append("--resume")
                else:
                    training_args.extend(["--run-id", train_run_id])
                _run(*training_args)
            training = json.loads(training_manifest.read_text(encoding="utf-8"))
            checkpoint = Path(training["selected_checkpoint"])
            if not _is_complete(deployment_manifest):
                deployment_base_args = [
                    scripts_dir / "evaluate_routed_deployment.py",
                    args.dataset,
                    args.split,
                    "--method",
                    args.method,
                    "--split-manifest",
                    split_path,
                    "--fold",
                    fold,
                    "--checkpoint",
                    checkpoint,
                    "--training-manifest",
                    training_manifest,
                    "--campaign-id",
                    args.campaign_id,
                    "--output-dir",
                    deployment_dir,
                    "--device",
                    args.device,
                    "--config-dir",
                    config_dir,
                ]
                deployment_state = (
                    json.loads(deployment_manifest.read_text(encoding="utf-8"))
                    if deployment_manifest.exists()
                    else {}
                )
                if not deployment_state.get("extraction_completed_at"):
                    extraction_args = [*deployment_base_args, "--stage", "extract"]
                    if deployment_manifest.exists():
                        extraction_args.extend(["--resume", deployment_run_id])
                    else:
                        extraction_args.extend(
                            ["--deployment-run-id", deployment_run_id]
                        )
                    _run(*extraction_args)
                qa_args = [
                    *deployment_base_args,
                    "--stage",
                    "qa",
                    "--resume",
                    deployment_run_id,
                ]
                if not _is_complete(deployment_manifest):
                    _run(*qa_args)
            aggregate_path = deployment_dir / "aggregate.json"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            manifest["fold_results"][str(fold)] = {
                "training_manifest": str(training_manifest),
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": training["selected_checkpoint_sha256"],
                "training_qa_usage": training.get("training_qa_usage", {}),
                "validation_qa_usage": training.get("validation_qa_usage", {}),
                "deployment_manifest": str(deployment_manifest),
                "aggregate": aggregate,
            }
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(manifest_path, manifest)
        campaign_usage = _campaign_usage(bundle.project.root_dir, args.campaign_id)
        cv_aggregate = _aggregate_folds(
            manifest["fold_results"], campaign_usage=campaign_usage
        )
        atomic_write_json(output_dir / "aggregate.json", cv_aggregate)
        _log_cv_summary(cv_aggregate, output_dir)
        manifest.update(
            status="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            aggregate=cv_aggregate,
        )
        atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(
            status="interrupted",
            last_error=f"{type(exc).__name__}: {exc}",
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(manifest_path, manifest)
        raise
    print(json.dumps(manifest["aggregate"], ensure_ascii=False, indent=2))


def _run(*arguments) -> None:
    command = [sys.executable, *[str(item) for item in arguments]]
    subprocess.run(command, check=True)


def _is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def _aggregate_folds(fold_results: dict, *, campaign_usage: dict) -> dict:
    folds = [item["aggregate"] for item in fold_results.values()]
    accuracy_stability = summarize_fold_accuracy(
        item["qa_accuracy_micro"] for item in folds
    )
    questions = sum(int(item["question_count"]) for item in folds)
    correct = sum(int(item["correct_count"]) for item in folds)
    sample_count = sum(int(item["sample_count"]) for item in folds)
    outcomes = [
        outcome for fold in folds for outcome in fold.get("question_outcomes", [])
    ]
    if len(outcomes) != questions:
        raise RuntimeError(
            "fold aggregates are missing classified question outcomes: "
            f"expected {questions}, got {len(outcomes)}"
        )
    question_metrics = summarize_question_outcomes(outcomes)
    extraction_cost = sum(
        float(item["memory_extraction"]["totals"]["known_cost"]) for item in folds
    )
    extraction_calls = sum(
        int(item["memory_extraction"]["totals"]["logical_api_calls"]) for item in folds
    )
    extraction_input = sum(
        int(item["memory_extraction"]["totals"]["input_tokens"]) for item in folds
    )
    extraction_output = sum(
        int(item["memory_extraction"]["totals"]["output_tokens"]) for item in folds
    )
    qa_usage = _aggregate_fold_qa_usage(folds)
    qa_cost = float(qa_usage["total_cost"])
    training_qa_cost = sum(
        float((item.get("training_qa_usage") or {}).get("total_cost", 0.0))
        + float((item.get("validation_qa_usage") or {}).get("total_cost", 0.0))
        for item in fold_results.values()
    )
    aggregate_metrics = {
        "overall": {
            "judge_correct": question_metrics["overall"]["judge_correct"]
        },
        **{
            category: {"judge_correct": metrics["judge_correct"]}
            for category, metrics in question_metrics["by_category"].items()
        },
    }
    category_distribution = {
        category.removeprefix("category_"): int(metrics["count"])
        for category, metrics in question_metrics["by_category"].items()
    }
    scope_values = {
        tuple(item.get("excluded_categories") or ()) for item in folds
    }
    if len(scope_values) != 1:
        raise RuntimeError("folds use inconsistent evaluation category exclusions")
    reader_models = {str(item.get("reader_model") or "") for item in folds}
    judge_models = {str(item.get("judge_model") or "") for item in folds}
    if len(reader_models) != 1 or len(judge_models) != 1:
        raise RuntimeError("folds use inconsistent Reader or Judge models")
    token_statistics = _qa_token_statistics(qa_usage)
    return {
        "schema_version": "routed_cv_aggregate_v2",
        "fold_count": len(folds),
        "sample_count": sample_count,
        "question_count": questions,
        "correct_count": correct,
        "qa_accuracy_micro": correct / questions if questions else 0.0,
        **accuracy_stability,
        "reader_model": reader_models.pop(),
        "judge_model": judge_models.pop(),
        "excluded_categories": list(scope_values.pop()),
        "question_outcomes": outcomes,
        "question_metrics": question_metrics,
        "category_distribution": category_distribution,
        "aggregate_metrics": aggregate_metrics,
        "qa_usage": qa_usage,
        "token_statistics": token_statistics,
        "memory_extraction": {
            "total_cost": extraction_cost,
            "logical_api_calls": extraction_calls,
            "input_tokens": extraction_input,
            "output_tokens": extraction_output,
            "cost_per_sample": extraction_cost / sample_count if sample_count else None,
            "amortized_cost_per_question": extraction_cost / questions if questions else None,
            "average_input_tokens_per_call": extraction_input / extraction_calls if extraction_calls else None,
            "average_output_tokens_per_call": extraction_output / extraction_calls if extraction_calls else None,
        },
        "training_candidate_extraction": campaign_usage,
        "qa_total_cost": qa_cost,
        "training_and_validation_qa_cost": training_qa_cost,
        "end_to_end_known_cost": extraction_cost + qa_cost,
        "full_experiment_known_cost": (
            float(campaign_usage["known_cost"])
            + training_qa_cost
            + extraction_cost
            + qa_cost
        ),
    }


def _aggregate_fold_qa_usage(folds: list[dict]) -> dict:
    fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "logical_api_calls",
        "reported_retry_count",
        "transport_attempts",
    )
    result = {"question_count": sum(int(item["question_count"]) for item in folds)}
    for role in ("reader", "judge"):
        result[role] = {
            field: sum(item["qa_usage"][role][field] for item in folds)
            for field in fields
        }
    for field in ("input_tokens", "output_tokens", "total_tokens", "input_cost", "output_cost", "total_cost", "logical_api_calls", "transport_attempts"):
        result[field] = result["reader"][field] + result["judge"][field]
    result["reader_cost"] = result["reader"]["total_cost"]
    result["judge_cost"] = result["judge"]["total_cost"]
    return result


def _qa_token_statistics(qa_usage: dict) -> dict:
    calls = int(qa_usage["logical_api_calls"])
    return {
        "scope": "qa_reader_plus_judge",
        "total_api_calls": calls,
        "total_prompt_tokens": int(qa_usage["input_tokens"]),
        "total_completion_tokens": int(qa_usage["output_tokens"]),
        "total_tokens": int(qa_usage["total_tokens"]),
        "avg_prompt_tokens_per_call": qa_usage["input_tokens"] / calls if calls else 0.0,
        "avg_completion_tokens_per_call": qa_usage["output_tokens"] / calls if calls else 0.0,
        "avg_total_tokens_per_call": qa_usage["total_tokens"] / calls if calls else 0.0,
        "reader": qa_usage["reader"],
        "judge": qa_usage["judge"],
    }


def _log_cv_summary(aggregate: dict, output_dir: Path) -> None:
    separator = "=" * 80
    logger.info(separator)
    logger.info("Cross-validation Evaluation Complete")
    logger.info(separator)
    logger.info("Total folds:      %d", aggregate["fold_count"])
    logger.info("Total samples:    %d", aggregate["sample_count"])
    logger.info("Total questions:  %d", aggregate["question_count"])
    logger.info("Reader model:     %s", aggregate["reader_model"])
    logger.info("Judge model:      %s", aggregate["judge_model"])
    if aggregate["excluded_categories"]:
        logger.info(
            "Excluded categories: %s",
            ", ".join(aggregate["excluded_categories"]),
        )
    logger.info("Category Distribution:")
    for metrics in aggregate["question_metrics"]["by_category"].values():
        logger.info(
            "  %s: %d questions (%.1f%%)",
            metrics["label"],
            metrics["count"],
            100.0 * metrics["fraction"],
        )
    logger.info("Aggregate Metrics:")
    groups = [("Overall", aggregate["question_metrics"]["overall"])]
    groups.extend(
        (item["label"], item)
        for item in aggregate["question_metrics"]["by_category"].values()
    )
    for label, metrics in groups:
        judge = metrics["judge_correct"]
        logger.info(
            "  %s judge_correct: mean=%.4f std=%.4f count=%d",
            label,
            judge["mean"],
            judge["std"],
            judge["count"],
        )
    token_stats = aggregate["token_statistics"]
    logger.info("Token Statistics (Reader + Judge):")
    logger.info("  Total API calls:         %d", token_stats["total_api_calls"])
    logger.info("  Total prompt tokens:     %s", f"{token_stats['total_prompt_tokens']:,}")
    logger.info(
        "  Total completion tokens: %s",
        f"{token_stats['total_completion_tokens']:,}",
    )
    logger.info("  Total tokens:            %s", f"{token_stats['total_tokens']:,}")
    logger.info("Results saved to: %s", output_dir / "aggregate.json")
    logger.info(separator)


def _campaign_usage(root: Path, campaign_id: str) -> dict:
    campaign_path = root / "outputs" / "rl_router" / "campaigns" / campaign_id / "manifest.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    summaries = []
    for run_id in campaign.get("expected_runs", {}).values():
        run_path = root / "outputs" / "rl_router" / "runs" / str(run_id) / "manifest.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        summaries.append(run.get("extraction_summary") or {})
    return {
        "campaign_id": campaign_id,
        "sample_count": len(summaries),
        "known_cost": sum(float(item.get("known_cost", 0.0)) for item in summaries),
        "unknown_cost_attempts": sum(
            int(item.get("unknown_cost_attempts", 0)) for item in summaries
        ),
        "logical_api_calls": sum(
            int((item.get("attempt_summary") or {}).get("logical_api_calls", 0))
            for item in summaries
        ),
        "input_tokens": sum(
            int((item.get("attempt_summary") or {}).get("provider_input_tokens", 0))
            for item in summaries
        ),
        "output_tokens": sum(
            int((item.get("attempt_summary") or {}).get("provider_output_tokens", 0))
            for item in summaries
        ),
    }


if __name__ == "__main__":
    main()

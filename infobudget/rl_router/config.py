"""Strict loader for the RL-router experiment configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from infobudget.config import ProjectBundle, load_project_bundle
from infobudget.rl_router.buffers import tier_config_int

REQUIRED_ROLES = {"small", "medium", "large", "qa_reader", "judge_llm"}
FACT_EXTRACTION_PROMPT_ROLES = {
    "locomo": "fact_extraction_locomo",
    "longmemeval": "fact_extraction_longmemeval",
}
SECRET_PATTERN = re.compile(r"(?:sk[-_][A-Za-z0-9_-]{12,}|api[_-]?key\s*:\s*['\"]?(?!\$|$).+)", re.I)


@dataclass(slots=True)
class RLConfigBundle:
    project: ProjectBundle
    embeddings: dict[str, dict[str, Any]]
    rl: dict[str, Any]
    config_dir: Path

    def prompt_path(self, role: str) -> Path:
        name = self.rl["prompts"][role]
        path = self.project.prompt_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required prompt is missing: {path}")
        return path

    def fact_extraction_prompt_role(self, dataset_name: str) -> str:
        normalized = str(dataset_name).strip().lower()
        try:
            return FACT_EXTRACTION_PROMPT_ROLES[normalized]
        except KeyError as exc:
            raise ValueError(
                f"unsupported fact-extraction dataset: {dataset_name!r}"
            ) from exc

    def fact_extraction_prompt_path(self, dataset_name: str) -> Path:
        return self.prompt_path(self.fact_extraction_prompt_role(dataset_name))

    def fact_extraction_prompt_version(self, dataset_name: str) -> str:
        role = self.fact_extraction_prompt_role(dataset_name)
        version = str((self.rl.get("prompt_versions") or {}).get(role) or "").strip()
        if not version:
            raise ValueError(f"prompt_versions.{role} must be a non-empty string")
        return version


def load_rl_bundle(config_dir: str | Path = "configs") -> RLConfigBundle:
    directory = Path(config_dir).resolve()
    project = load_project_bundle(directory)
    missing_roles = REQUIRED_ROLES - project.models.keys()
    if missing_roles:
        raise ValueError(f"models.yaml is missing roles: {sorted(missing_roles)}")
    for role in REQUIRED_ROLES:
        spec = project.models[role]
        if spec.deploy == "api" and not spec.api_key_env:
            raise ValueError(f"model role {role} must configure api_key_env")
        if spec.model_name not in project.prices:
            raise ValueError(f"prices.yaml has no price snapshot for {spec.model_name}")
    embeddings = _read_yaml(directory / "embeddings.yaml").get("embeddings", {})
    rl = _read_yaml(directory / "rl_router.yaml")
    model_family = str(rl.get("model_family") or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", model_family):
        raise ValueError(
            "model_family must be a non-empty lowercase namespace component"
        )
    rl["model_family"] = model_family
    known_family_markers = {"qwen": "qwen", "llama": "llama"}
    marker = known_family_markers.get(model_family)
    if marker is not None:
        mismatched_roles = [
            role
            for role in ("small", "medium", "large")
            if marker not in project.models[role].effective_model_name.casefold()
        ]
        if mismatched_roles:
            raise ValueError(
                f"model_family={model_family} does not match extraction model roles: "
                + ", ".join(mismatched_roles)
            )
    extraction = rl["extraction"]
    if not isinstance(extraction.get("allow_oversize_singleton", False), bool):
        raise ValueError("extraction.allow_oversize_singleton must be a boolean")
    if not isinstance(extraction.get("truncate_over_total_context", False), bool):
        raise ValueError("extraction.truncate_over_total_context must be a boolean")
    repair_attempts = int(extraction.get("schema_repair_max_attempts", 2))
    if repair_attempts < 0 or repair_attempts > 5:
        raise ValueError("extraction.schema_repair_max_attempts must be between 0 and 5")
    max_facts = extraction.get("max_facts_per_segment")
    max_fact_values = max_facts.values() if isinstance(max_facts, dict) else (max_facts,)
    if any(value is None or int(value) <= 0 for value in max_fact_values):
        raise ValueError("extraction.max_facts_per_segment must be positive")
    quality_gates = extraction.get("quality_gates", {})
    for key in (
        "max_empty_fact_segment_rate",
        "max_saturated_segment_rate",
        "max_repair_batch_rate",
        "max_failed_batch_rate",
    ):
        value = float(quality_gates.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"extraction.quality_gates.{key} must be between 0 and 1")
    for tier in ("small", "medium", "large"):
        spec = project.models[tier]
        if spec.max_output_tokens <= 0 or spec.max_output_tokens >= spec.max_context_tokens:
            raise ValueError(
                f"model role {tier} must configure max_output_tokens below max_context_tokens"
            )
        buffer_cfg = extraction["buffers"][tier]
        max_segments = int(buffer_cfg["max_segments"])
        max_input = int(buffer_cfg["max_input_tokens"])
        max_total = int(buffer_cfg["max_total_context_tokens"])
        reserve = tier_config_int(extraction, "reserve_output_tokens_per_segment", tier)
        if max_segments <= 0 or max_input <= 0 or max_total <= 0 or reserve <= 0:
            raise ValueError(f"extraction buffer {tier} limits must be positive")
        if max_input > spec.max_input_tokens:
            raise ValueError(
                f"extraction buffer {tier} max_input_tokens exceeds model input capacity"
            )
        if max_total > spec.max_context_tokens or max_input > max_total:
            raise ValueError(f"extraction buffer {tier} exceeds model context capacity")
        if reserve * max_segments > spec.max_output_tokens:
            raise ValueError(
                f"extraction buffer {tier} reserved output exceeds model max_output_tokens"
            )
    candidate_model_ids = [
        project.models[tier].stable_model_id for tier in ("small", "medium", "large")
    ]
    if any(not model_id for model_id in candidate_model_ids):
        raise ValueError("candidate model_id values must be non-empty")
    if len(candidate_model_ids) != len(set(candidate_model_ids)):
        raise ValueError("small, medium, and large must use distinct model_id values")
    reliability = rl.get("api_reliability", {})
    if int(reliability.get("timeout_seconds", 120)) <= 0:
        raise ValueError("api_reliability.timeout_seconds must be positive")
    if not 0 <= int(reliability.get("max_retries", 3)) <= 10:
        raise ValueError("api_reliability.max_retries must be between 0 and 10")
    if float(reliability.get("retry_backoff_seconds", 1.0)) < 0:
        raise ValueError("api_reliability.retry_backoff_seconds cannot be negative")
    storage = rl.get("storage", {})
    namespace_template = str(storage.get("collection_namespace") or "")
    required_namespace_fields = {
        "{project_name}",
        "{model_family}",
        "{dataset}",
        "{split}",
        "{segmentation_version}",
        "{embedding_hash}",
    }
    missing_namespace_fields = sorted(
        field for field in required_namespace_fields if field not in namespace_template
    )
    if missing_namespace_fields:
        raise ValueError(
            "storage.collection_namespace is missing required placeholders: "
            + ", ".join(missing_namespace_fields)
        )
    storage_mode = str(storage.get("mode", "")).strip().lower()
    if storage_mode not in {"local", "server"}:
        raise ValueError("storage.mode must be local or server")
    if int(storage.get("vector_size", 0)) <= 0:
        raise ValueError("storage.vector_size must be positive")
    if storage_mode == "local" and not storage.get("local_path"):
        raise ValueError("storage.local_path is required in local mode")
    if storage_mode == "server":
        url = str(storage.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("storage.url must be an http(s) Qdrant server URL")
        if parsed.username or parsed.password:
            raise ValueError("storage.url must not contain credentials; use storage.api_key_env")
        if float(storage.get("timeout_seconds", 30)) <= 0:
            raise ValueError("storage.timeout_seconds must be positive")
        if not 1 <= int(storage.get("grpc_port", 6334)) <= 65535:
            raise ValueError("storage.grpc_port must be between 1 and 65535")
        api_key_env = storage.get("api_key_env", "")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise ValueError("storage.api_key_env must be a string")
    for role in ("router", "memory"):
        if role not in embeddings:
            raise ValueError(f"embeddings.yaml is missing {role}")
        if not embeddings[role].get("local_files_only", False):
            raise ValueError(f"embedding role {role} must set local_files_only=true")
        if int(embeddings[role].get("dimension", 0)) <= 0:
            raise ValueError(f"embedding role {role} dimension must be positive")
        if embeddings[role].get("long_text_strategy", "truncate") not in {
            "truncate",
            "mean_pool_chunks",
        }:
            raise ValueError(
                f"embedding role {role} has invalid long_text_strategy"
            )
    if int(storage["vector_size"]) != int(embeddings["memory"]["dimension"]):
        raise ValueError(
            "storage.vector_size must equal embeddings.memory.dimension"
        )
    evaluation = rl["evaluation"]
    for key, model_role in (("reader_max_new_tokens", "qa_reader"), ("judge_max_new_tokens", "judge_llm")):
        limit = int(evaluation[key])
        if limit <= 0 or limit > project.models[model_role].max_output_tokens:
            raise ValueError(f"evaluation.{key} exceeds the configured {model_role} output limit")
    if float(evaluation.get("judge_temperature", 0.0)) != 0.0:
        raise ValueError("LightMEM judge_temperature must remain 0.0")
    exclusions = evaluation.get("excluded_categories_by_dataset", {})
    if not isinstance(exclusions, dict):
        raise ValueError(
            "evaluation.excluded_categories_by_dataset must be a mapping"
        )
    unknown_datasets = sorted(set(exclusions) - {"locomo", "longmemeval"})
    if unknown_datasets:
        raise ValueError(
            "evaluation.excluded_categories_by_dataset has unsupported datasets: "
            + ", ".join(unknown_datasets)
        )
    for dataset_name, values in exclusions.items():
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(
                "evaluation.excluded_categories_by_dataset."
                f"{dataset_name} must be a list of non-empty strings"
            )
    router = rl["router"]
    if router.get("type") != "embedding_mlp":
        raise ValueError("training CLI currently requires router.type=embedding_mlp")
    if router.get("algorithm") != "actor_critic_lagrangian":
        raise ValueError("training CLI currently requires actor_critic_lagrangian")
    if list(router.get("actions") or []) != ["small", "medium", "large"]:
        raise ValueError("router.actions must be [small, medium, large] in that order")
    budget = float(router["budget"])
    if not 0.0 <= budget <= 1.0:
        raise ValueError("normalized router.budget must be between 0 and 1")
    if any(int(value) <= 0 for value in router["hidden_dimensions"]):
        raise ValueError("router.hidden_dimensions must be positive")
    if float(router["learning_rate"]) <= 0 or float(router["lambda_learning_rate"]) <= 0:
        raise ValueError("router learning rates must be positive")
    if float(router.get("value_loss_coefficient", 0.5)) < 0:
        raise ValueError("router.value_loss_coefficient cannot be negative")
    if float(router.get("entropy_coefficient", 0.01)) < 0:
        raise ValueError("router.entropy_coefficient cannot be negative")
    if float(router.get("max_gradient_norm", 1.0)) <= 0:
        raise ValueError("router.max_gradient_norm must be positive")
    if float(router.get("gamma", 1.0)) != 1.0:
        raise ValueError("the current contextual-bandit trainer requires router.gamma=1.0")
    bundle = RLConfigBundle(project, embeddings, rl, directory)
    for prompt_role in rl.get("prompts", {}):
        bundle.prompt_path(prompt_role)
    for dataset_name in FACT_EXTRACTION_PROMPT_ROLES:
        bundle.fact_extraction_prompt_path(dataset_name)
        bundle.fact_extraction_prompt_version(dataset_name)
    return bundle


def scan_config_secrets(config_dir: str | Path) -> list[Path]:
    findings: list[Path] = []
    for path in Path(config_dir).glob("*.yaml"):
        if SECRET_PATTERN.search(path.read_text(encoding="utf-8")):
            findings.append(path)
    return findings


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value

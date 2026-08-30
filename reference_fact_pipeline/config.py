"""Configuration loader for the frozen-reference pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ReferencePipelineConfig:
    schema_version: str
    prompt_version: str
    reference_extractor_role: str
    coverage_extractor_role: str
    grounding_judge_role: str
    candidate_roles: tuple[str, ...]
    require_non_candidate_reference_model: bool
    max_reference_facts: int
    max_raw_facts: int
    extraction_max_new_tokens: int
    coverage_max_new_tokens: int
    grounding_max_new_tokens: int
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float
    cloudflare_request_interval_seconds: float
    cloudflare_402_backoff_seconds: tuple[float, ...]
    cloudflare_circuit_pause_seconds: float
    cloudflare_max_auto_resumes: int
    fact_type_priority: tuple[str, ...]

    def canonical_hash(self) -> str:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def load_reference_config(path: str | Path) -> ReferencePipelineConfig:
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"reference config root must be an object: {source}")
    models = _object(value, "models")
    limits = _object(value, "limits")
    api = _object(value, "api")
    ranking = _object(value, "ranking")
    config = ReferencePipelineConfig(
        schema_version=str(value.get("schema_version") or "").strip(),
        prompt_version=str(value.get("prompt_version") or "").strip(),
        reference_extractor_role=str(models.get("reference_extractor_role") or "").strip(),
        coverage_extractor_role=str(models.get("coverage_extractor_role") or "").strip(),
        grounding_judge_role=str(models.get("grounding_judge_role") or "").strip(),
        candidate_roles=tuple(str(item) for item in models.get("candidate_roles", ())),
        require_non_candidate_reference_model=bool(
            models.get("require_non_candidate_reference_model", True)
        ),
        max_reference_facts=int(limits.get("max_reference_facts", 15)),
        max_raw_facts=int(limits.get("max_raw_facts", 45)),
        extraction_max_new_tokens=int(limits.get("extraction_max_new_tokens", 4096)),
        coverage_max_new_tokens=int(limits.get("coverage_max_new_tokens", 4096)),
        grounding_max_new_tokens=int(limits.get("grounding_max_new_tokens", 4096)),
        timeout_seconds=int(api.get("timeout_seconds", 120)),
        max_retries=int(api.get("max_retries", 3)),
        retry_backoff_seconds=float(api.get("retry_backoff_seconds", 1.0)),
        cloudflare_request_interval_seconds=float(
            api.get("cloudflare_request_interval_seconds", 3.0)
        ),
        cloudflare_402_backoff_seconds=tuple(
            float(item)
            for item in api.get("cloudflare_402_backoff_seconds", (60, 120, 300))
        ),
        cloudflare_circuit_pause_seconds=float(
            api.get("cloudflare_circuit_pause_seconds", 300.0)
        ),
        cloudflare_max_auto_resumes=int(api.get("cloudflare_max_auto_resumes", 2)),
        fact_type_priority=tuple(str(item) for item in ranking.get("fact_type_priority", ())),
    )
    _validate(config)
    return config


def _object(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key, {})
    if not isinstance(item, dict):
        raise ValueError(f"reference config field {key} must be an object")
    return item


def _validate(config: ReferencePipelineConfig) -> None:
    required = {
        "schema_version": config.schema_version,
        "prompt_version": config.prompt_version,
        "reference_extractor_role": config.reference_extractor_role,
        "coverage_extractor_role": config.coverage_extractor_role,
        "grounding_judge_role": config.grounding_judge_role,
    }
    empty = [name for name, value in required.items() if not value]
    if empty:
        raise ValueError(f"empty reference config fields: {', '.join(empty)}")
    for name in (
        "max_reference_facts",
        "max_raw_facts",
        "extraction_max_new_tokens",
        "coverage_max_new_tokens",
        "grounding_max_new_tokens",
        "timeout_seconds",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.max_raw_facts < config.max_reference_facts:
        raise ValueError("max_raw_facts must be >= max_reference_facts")
    if config.max_retries < 0 or config.retry_backoff_seconds < 0:
        raise ValueError("API retry values cannot be negative")
    if config.cloudflare_request_interval_seconds < 0:
        raise ValueError("cloudflare_request_interval_seconds cannot be negative")
    if not config.cloudflare_402_backoff_seconds or any(
        value < 0 for value in config.cloudflare_402_backoff_seconds
    ):
        raise ValueError("cloudflare_402_backoff_seconds must be a non-empty non-negative list")
    if config.cloudflare_circuit_pause_seconds < 0:
        raise ValueError("cloudflare_circuit_pause_seconds cannot be negative")
    if config.cloudflare_max_auto_resumes < 0:
        raise ValueError("cloudflare_max_auto_resumes cannot be negative")

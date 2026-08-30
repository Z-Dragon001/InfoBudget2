"""Two-pass construction of strong, candidate-independent frozen reference Facts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from infobudget.rl_router.api import ChatCompletionClient, LLMResponse
from infobudget.rl_router.ledger import atomic_write_json
from infobudget.rl_router.schemas import TopicSegment
from infobudget.schemas import ModelSpec, PriceSpec
from reference_fact_pipeline.config import ReferencePipelineConfig
from reference_fact_pipeline.parsing import (
    parse_grounding_decisions,
    parse_proposed_facts,
)
from reference_fact_pipeline.schemas import (
    FrozenReferenceFact,
    FrozenReferenceSet,
    GroundingDecision,
    ProposedFact,
    StageUsage,
)


class ReferenceFactPipeline:
    def __init__(
        self,
        *,
        config: ReferencePipelineConfig,
        models: dict[str, ModelSpec],
        prices: dict[str, PriceSpec],
        client: ChatCompletionClient,
        prompt_dir: str | Path,
        candidate_prompt_dir: str | Path = "configs/prompts",
        raw_archive_dir: str | Path | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config
        self.models = models
        self.prices = prices
        self.client = client
        self.prompt_dir = Path(prompt_dir)
        self.candidate_prompt_dir = Path(candidate_prompt_dir)
        self.raw_archive_dir = Path(raw_archive_dir) if raw_archive_dir else None
        self.progress_callback = progress_callback
        self._prompts = {
            name: (self.prompt_dir / filename).read_text(encoding="utf-8")
            for name, filename in {
                "locomo": "locomo_extract.txt",
                "longmemeval": "longmemeval_extract.txt",
                "coverage": "coverage.txt",
                "grounding": "grounding.txt",
                "json_repair": "json_repair.txt",
            }.items()
        }
        self._fact_policies = {
            dataset: _shared_fact_policy(
                self.candidate_prompt_dir / filename,
                max_raw_facts=self.config.max_raw_facts,
            )
            for dataset, filename in {
                "locomo": "locomo_memory_extraction.txt",
                "longmemeval": "longmemeval_memory_extraction.txt",
            }.items()
        }
        self.effective_config_hash = _effective_config_hash(
            self.config.canonical_hash(), self._prompts, self._fact_policies
        )
        self._validate_models()

    def process_segment(self, segment: TopicSegment, *, run_id: str) -> FrozenReferenceSet:
        dataset = segment.dataset_name.lower()
        if dataset not in {"locomo", "longmemeval"}:
            raise ValueError(f"unsupported reference dataset: {segment.dataset_name}")
        turn_ids = set(segment.turn_ids)
        usage: list[StageUsage] = []

        extraction_prompt = self._render(
            self._prompts[dataset],
            segment=segment,
            shared_fact_policy=self._fact_policies[dataset],
            max_raw_facts=str(self.config.max_raw_facts),
        )
        extraction_response, extraction_usage = self._complete(
            stage="initial_extraction",
            role=self.config.reference_extractor_role,
            prompt=extraction_prompt,
            max_new_tokens=self.config.extraction_max_new_tokens,
        )
        usage.append(extraction_usage)
        self._archive(segment, run_id, "initial_extraction", extraction_prompt, extraction_response)
        try:
            initial = parse_proposed_facts(
                extraction_response.content,
                segment_turn_ids=turn_ids,
                origin="initial",
                max_facts=self.config.max_raw_facts,
            )
        except ValueError as error:
            repaired, repair_usage = self._repair_json(
                segment=segment,
                run_id=run_id,
                stage="initial_extraction_json_repair",
                role=self.config.reference_extractor_role,
                invalid_output=extraction_response.content,
                validation_error=error,
                expected_field="facts",
                max_items=self.config.max_raw_facts,
                max_new_tokens=self.config.extraction_max_new_tokens,
            )
            usage.append(repair_usage)
            initial = parse_proposed_facts(
                repaired.content,
                segment_turn_ids=turn_ids,
                origin="initial",
                max_facts=self.config.max_raw_facts,
            )

        accepted_initial, rejected_initial, grounding_usage = self._ground(
            segment,
            initial,
            existing=(),
            stage="initial_grounding",
            run_id=run_id,
        )
        usage.extend(grounding_usage)

        coverage_prompt = self._render(
            self._prompts["coverage"],
            segment=segment,
            accepted_facts=json.dumps(
                [item.to_dict() for item, _ in accepted_initial],
                ensure_ascii=False,
                sort_keys=True,
            ),
            shared_fact_policy=self._fact_policies[dataset],
            max_raw_facts=str(self.config.max_raw_facts),
        )
        coverage_response, coverage_usage = self._complete(
            stage="coverage_extraction",
            role=self.config.coverage_extractor_role,
            prompt=coverage_prompt,
            max_new_tokens=self.config.coverage_max_new_tokens,
        )
        usage.append(coverage_usage)
        remaining = max(0, self.config.max_raw_facts - len(initial))
        self._archive(segment, run_id, "coverage_extraction", coverage_prompt, coverage_response)
        try:
            coverage = parse_proposed_facts(
                coverage_response.content,
                segment_turn_ids=turn_ids,
                origin="coverage",
                max_facts=remaining,
            )
        except ValueError as error:
            repaired, repair_usage = self._repair_json(
                segment=segment,
                run_id=run_id,
                stage="coverage_extraction_json_repair",
                role=self.config.coverage_extractor_role,
                invalid_output=coverage_response.content,
                validation_error=error,
                expected_field="missing_facts",
                max_items=remaining,
                max_new_tokens=self.config.coverage_max_new_tokens,
            )
            usage.append(repair_usage)
            coverage = parse_proposed_facts(
                repaired.content,
                segment_turn_ids=turn_ids,
                origin="coverage",
                max_facts=remaining,
            )

        accepted_coverage, rejected_coverage, coverage_grounding_usage = self._ground(
            segment,
            coverage,
            existing=accepted_initial,
            stage="coverage_grounding",
            run_id=run_id,
        )
        usage.extend(coverage_grounding_usage)

        accepted = self._deduplicate([*accepted_initial, *accepted_coverage])
        ranked = sorted(accepted, key=self._rank_key)
        selected = ranked[: self.config.max_reference_facts]
        frozen = [
            FrozenReferenceFact(
                reference_fact_id=self._reference_fact_id(segment, fact),
                fact_text=fact.fact_text,
                source_turn_ids=fact.source_turn_ids,
                fact_type=fact.fact_type,
                state_status=fact.state_status,
                origin=fact.origin,
                grounding_reason=decision.reason,
                selection_rank=index,
            )
            for index, (fact, decision) in enumerate(selected, start=1)
        ]
        reference_hash = self._reference_set_hash(frozen)
        return FrozenReferenceSet(
            schema_version=self.config.schema_version,
            dataset_name=segment.dataset_name,
            split=segment.split,
            sample_id=segment.sample_id,
            session_id=segment.session_id,
            segment_id=segment.segment_id,
            segment_order=segment.segment_order,
            segmentation_method=segment.segmentation_method,
            segmentation_version=segment.segmentation_version,
            source_content_hash=segment.source_content_hash,
            segment_turn_ids=segment.turn_ids,
            reference_set_hash=reference_hash,
            reference_facts=frozen,
            rejected_facts=[*rejected_initial, *rejected_coverage],
            raw_proposal_count=len(initial) + len(coverage),
            grounded_accept_count=len(accepted),
            frozen_fact_count=len(frozen),
            truncated_to_k=len(ranked) > self.config.max_reference_facts,
            run_id=run_id,
            prompt_version=self.config.prompt_version,
            config_hash=self.effective_config_hash,
            stage_usage=usage,
        )

    def _ground(
        self,
        segment: TopicSegment,
        proposed: list[ProposedFact],
        *,
        existing: Iterable[tuple[ProposedFact, GroundingDecision]],
        stage: str,
        run_id: str,
    ) -> tuple[
        list[tuple[ProposedFact, GroundingDecision]],
        list[dict[str, Any]],
        list[StageUsage],
    ]:
        if not proposed:
            return [], [], []
        existing_list = list(existing)
        prompt = self._render(
            self._prompts["grounding"],
            segment=segment,
            proposed_facts=json.dumps(
                [item.to_dict() for item in proposed], ensure_ascii=False, sort_keys=True
            ),
            existing_facts=json.dumps(
                [item.to_dict() for item, _ in existing_list],
                ensure_ascii=False,
                sort_keys=True,
            ),
            shared_fact_policy=self._fact_policies[segment.dataset_name.lower()],
            max_raw_facts=str(self.config.max_raw_facts),
        )
        response, usage = self._complete(
            stage=stage,
            role=self.config.grounding_judge_role,
            prompt=prompt,
            max_new_tokens=self.config.grounding_max_new_tokens,
        )
        self._archive(segment, run_id, stage, prompt, response)
        stage_usage = [usage]
        try:
            decisions = parse_grounding_decisions(response.content, proposed)
        except ValueError as error:
            repaired, repair_usage = self._repair_json(
                segment=segment,
                run_id=run_id,
                stage=f"{stage}_json_repair",
                role=self.config.grounding_judge_role,
                invalid_output=response.content,
                validation_error=error,
                expected_field="decisions",
                max_items=len(proposed),
                max_new_tokens=self.config.grounding_max_new_tokens,
            )
            stage_usage.append(repair_usage)
            decisions = parse_grounding_decisions(repaired.content, proposed)
        accepted: list[tuple[ProposedFact, GroundingDecision]] = []
        rejected: list[dict[str, Any]] = []
        for fact in proposed:
            decision = decisions[fact.temp_fact_id]
            if decision.accepted:
                accepted.append((fact, decision))
            else:
                rejected.append({"fact": fact.to_dict(), "decision": asdict(decision)})
        return accepted, rejected, stage_usage

    def _repair_json(
        self,
        *,
        segment: TopicSegment,
        run_id: str,
        stage: str,
        role: str,
        invalid_output: str,
        validation_error: Exception,
        expected_field: str,
        max_items: int,
        max_new_tokens: int,
    ) -> tuple[LLMResponse, StageUsage]:
        prompt = self._prompts["json_repair"]
        replacements = {
            "segment_id": segment.segment_id,
            "valid_source_turn_ids": json.dumps(list(segment.turn_ids)),
            "expected_field": expected_field,
            "required_schema": json.dumps(
                _repair_schema(
                    segment.segment_id,
                    expected_field,
                    segment.turn_ids[0],
                    max_items,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            "max_items": str(max_items),
            "validation_error": str(validation_error)[:2000],
            "invalid_output": invalid_output,
        }
        for name, value in replacements.items():
            prompt = prompt.replace("{" + name + "}", value)
        response, usage = self._complete(
            stage=stage,
            role=role,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        self._archive(segment, run_id, stage, prompt, response)
        return response, usage

    def _complete(
        self, *, stage: str, role: str, prompt: str, max_new_tokens: int
    ) -> tuple[LLMResponse, StageUsage]:
        model = self.models[role]
        self._notify_progress({"event": "stage_started", "stage": stage, "role": role})
        try:
            response = self.client.complete(
                model_spec=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                json_mode=True,
            )
        except Exception as error:
            self._notify_progress(
                {
                    "event": "stage_failed",
                    "stage": stage,
                    "role": role,
                    "error": str(error)[:500],
                }
            )
            raise
        price = self.prices.get(model.model_name) or self.prices.get(model.effective_model_name)
        input_cost = (
            response.input_tokens * price.official_price_in_per_1m / 1_000_000
            if price is not None
            else 0.0
        )
        output_cost = (
            response.output_tokens * price.official_price_out_per_1m / 1_000_000
            if price is not None
            else 0.0
        )
        usage = StageUsage(
            stage=stage,
            role=role,
            configured_model=model.model_name,
            request_model=model.effective_model_name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            usage_source=response.usage_source,
            retry_count=response.retry_count,
            latency_ms=response.latency_ms,
            provider_request_id=response.provider_request_id,
            finish_reason=response.finish_reason,
            cost_status="known" if price is not None else "unknown_missing_price_snapshot",
        )
        self._notify_progress(
            {
                "event": "stage_completed",
                "stage": stage,
                "role": role,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
            }
        )
        return response, usage

    def _notify_progress(self, event: dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            return

    def _render(self, template: str, *, segment: TopicSegment, **values: str) -> str:
        replacements = {
            "dataset_name": segment.dataset_name,
            "segment_id": segment.segment_id,
            "turn_ids": json.dumps(list(segment.turn_ids)),
            "segment_text": _source_annotated_text(segment),
            **values,
        }
        rendered = template
        for name, value in replacements.items():
            rendered = rendered.replace("{" + name + "}", value)
        return rendered

    def _rank_key(
        self, item: tuple[ProposedFact, GroundingDecision]
    ) -> tuple[int, int, int, int, str]:
        fact, _ = item
        priorities = {name: index for index, name in enumerate(self.config.fact_type_priority)}
        current_rank = {"current": 0, "timeless": 1, "historical": 2, "unspecified": 3}
        return (
            priorities.get(fact.fact_type, len(priorities)),
            current_rank.get(fact.state_status, 3),
            min(fact.source_turn_ids),
            fact.proposal_order,
            _normalize_fact(fact.fact_text),
        )

    @staticmethod
    def _deduplicate(
        facts: list[tuple[ProposedFact, GroundingDecision]],
    ) -> list[tuple[ProposedFact, GroundingDecision]]:
        seen: set[tuple[str, tuple[int, ...]]] = set()
        result: list[tuple[ProposedFact, GroundingDecision]] = []
        for item in facts:
            fact, _ = item
            key = (_normalize_fact(fact.fact_text), fact.source_turn_ids)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _reference_fact_id(segment: TopicSegment, fact: ProposedFact) -> str:
        payload = json.dumps(
            [segment.segment_id, _normalize_fact(fact.fact_text), list(fact.source_turn_ids)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "rf_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _reference_set_hash(facts: list[FrozenReferenceFact]) -> str:
        payload = [
            [item.reference_fact_id, item.fact_text, list(item.source_turn_ids)]
            for item in facts
        ]
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _archive(
        self,
        segment: TopicSegment,
        run_id: str,
        stage: str,
        prompt: str,
        response: LLMResponse,
    ) -> None:
        if self.raw_archive_dir is None:
            return
        safe_segment = hashlib.sha256(segment.segment_id.encode("utf-8")).hexdigest()[:16]
        atomic_write_json(
            self.raw_archive_dir / safe_segment / f"{stage}.json",
            {
                "run_id": run_id,
                "segment_id": segment.segment_id,
                "stage": stage,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "response": response.content,
                "usage": {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "usage_source": response.usage_source,
                    "retry_count": response.retry_count,
                    "latency_ms": response.latency_ms,
                    "attempts": response.attempts or [],
                },
            },
        )

    def _validate_models(self) -> None:
        roles = {
            self.config.reference_extractor_role,
            self.config.coverage_extractor_role,
            self.config.grounding_judge_role,
            *self.config.candidate_roles,
        }
        missing = sorted(roles - self.models.keys())
        if missing:
            raise ValueError(f"unknown model roles in reference config: {', '.join(missing)}")
        if not self.config.require_non_candidate_reference_model:
            return
        candidate_names = {
            self.models[role].effective_model_name.lower()
            for role in self.config.candidate_roles
        }
        for role in (
            self.config.reference_extractor_role,
            self.config.coverage_extractor_role,
            self.config.grounding_judge_role,
        ):
            if self.models[role].effective_model_name.lower() in candidate_names:
                raise ValueError(
                    f"reference role {role} uses a candidate model; this leaks candidate bias"
                )


def _normalize_fact(text: str) -> str:
    return " ".join(text.casefold().strip().rstrip("。.!！?").split())


def _repair_schema(
    segment_id: str,
    expected_field: str,
    example_source_turn_id: int,
    max_items: int,
) -> dict[str, Any]:
    if expected_field in {"facts", "missing_facts"}:
        item: dict[str, Any] = {
            "fact_text": "Preserve the original Fact text",
            "source_turn_ids": [example_source_turn_id],
            "fact_type": "Preserve the original allowed value",
            "state_status": "Preserve the original allowed value",
        }
    elif expected_field == "decisions":
        item = {
            "temp_fact_id": "Preserve the original proposal ID",
            "decision": "ACCEPT|REJECT",
            "entailed": True,
            "atomic": True,
            "source_ids_sufficient": True,
            "contains_external_inference": False,
            "duplicate_of": None,
            "reason": "Preserve the original reason",
        }
    else:
        raise ValueError(f"unsupported JSON repair field: {expected_field}")
    return {"segment_id": segment_id, expected_field: [item] if max_items > 0 else []}


def _shared_fact_policy(path: Path, *, max_raw_facts: int) -> str:
    """Reuse the candidate prompt's semantic policy without its output contract."""
    if not path.is_file():
        raise FileNotFoundError(f"candidate Fact prompt is missing: {path}")
    text = path.read_text(encoding="utf-8")
    start_marker = "Mandatory processing procedure\n"
    end_marker = "Output contract\n"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end <= start:
        raise ValueError(f"candidate prompt lacks policy section markers: {path}")
    policy = text[start:end].strip()
    policy = policy.replace(
        "Maximum output: 15 facts per topic segment.",
        "Final frozen output: 15 facts per topic segment. The high-recall proposal "
        f"stage may return up to {max_raw_facts} proposals before grounding and final selection.",
    )
    policy = policy.replace(
        "If a segment supports more than 15 candidate facts, retain the 15 facts",
        "When the final grounded set supports more than 15 facts, retain the 15 facts",
    )
    policy = policy.replace(
        "Do not exceed the limit. Reduce redundancy before dropping distinct information.",
        "Apply this priority order to the proposal ordering and final deterministic selection. "
        "Reduce redundancy before dropping distinct information.",
    )
    policy = policy.replace(
        "Do not exceed the limit. Remove redundancy before dropping a distinct fact.",
        "Apply this priority order to the proposal ordering and final deterministic selection. "
        "Remove redundancy before dropping a distinct fact.",
    )
    return policy


def _effective_config_hash(
    config_hash: str,
    prompts: dict[str, str],
    policies: dict[str, str],
) -> str:
    payload = {
        "config_hash": config_hash,
        "reference_prompts": prompts,
        "shared_candidate_policies": policies,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _source_annotated_text(segment: TopicSegment) -> str:
    """Render one unambiguous canonical ID and remove the legacy 0-based label."""
    # A LongMemEval turn may contain paragraphs or example dialogue. Find only
    # the expected sequential legacy header (canonical turn_id - 1), preventing
    # numbered lists/example messages inside one turn from becoming boundaries.
    if not segment.turn_ids:
        raise ValueError(f"segment has no source turns: {segment.segment_id}")
    if segment.start_turn != segment.turn_ids[0] or segment.end_turn != segment.turn_ids[-1]:
        raise ValueError(f"segment turn bounds do not match turn_ids: {segment.segment_id}")
    headers: list[re.Match[str]] = []
    cursor = 0
    for turn_id in segment.turn_ids:
        header = re.compile(
            rf"(?m)^\[(?P<timestamp>[^\]\r\n]+)\]\s+"
            rf"{int(turn_id) - 1}\.(?P<speaker>[^:\r\n]+):\s*"
        )
        match = header.search(segment.text, cursor)
        if match is None:
            raise ValueError(
                f"cannot locate source turn {turn_id} in segment text: {segment.segment_id}"
            )
        headers.append(match)
        cursor = match.end()
    if headers[0].start() != 0:
        raise ValueError(f"segment text has an unrecognized prefix: {segment.segment_id}")
    rendered: list[str] = []
    for index, (turn_id, header) in enumerate(
        zip(segment.turn_ids, headers, strict=True)
    ):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(segment.text)
        content = segment.text[header.end() : end].rstrip("\r\n")
        rendered.append(
            f"<SOURCE_TURN_ID={turn_id}> [{header.group('timestamp')}] "
            f"{header.group('speaker').strip()}: {content}"
        )
    return "\n".join(rendered)

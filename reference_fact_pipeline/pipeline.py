"""Two-pass construction of strong, candidate-independent frozen reference Facts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

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
        raw_archive_dir: str | Path | None = None,
    ):
        self.config = config
        self.models = models
        self.prices = prices
        self.client = client
        self.prompt_dir = Path(prompt_dir)
        self.raw_archive_dir = Path(raw_archive_dir) if raw_archive_dir else None
        self._prompts = {
            name: (self.prompt_dir / filename).read_text(encoding="utf-8")
            for name, filename in {
                "locomo": "locomo_extract.txt",
                "longmemeval": "longmemeval_extract.txt",
                "coverage": "coverage.txt",
                "grounding": "grounding.txt",
            }.items()
        }
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
        )
        extraction_response, extraction_usage = self._complete(
            stage="initial_extraction",
            role=self.config.reference_extractor_role,
            prompt=extraction_prompt,
            max_new_tokens=self.config.extraction_max_new_tokens,
        )
        usage.append(extraction_usage)
        initial = parse_proposed_facts(
            extraction_response.content,
            segment_turn_ids=turn_ids,
            origin="initial",
            max_facts=self.config.max_raw_facts,
        )
        self._archive(segment, run_id, "initial_extraction", extraction_prompt, extraction_response)

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
        )
        coverage_response, coverage_usage = self._complete(
            stage="coverage_extraction",
            role=self.config.coverage_extractor_role,
            prompt=coverage_prompt,
            max_new_tokens=self.config.coverage_max_new_tokens,
        )
        usage.append(coverage_usage)
        remaining = max(0, self.config.max_raw_facts - len(initial))
        coverage = parse_proposed_facts(
            coverage_response.content,
            segment_turn_ids=turn_ids,
            origin="coverage",
            max_facts=remaining,
        )
        self._archive(segment, run_id, "coverage_extraction", coverage_prompt, coverage_response)

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
            config_hash=self.config.canonical_hash(),
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
        )
        response, usage = self._complete(
            stage=stage,
            role=self.config.grounding_judge_role,
            prompt=prompt,
            max_new_tokens=self.config.grounding_max_new_tokens,
        )
        self._archive(segment, run_id, stage, prompt, response)
        decisions = parse_grounding_decisions(response.content, proposed)
        accepted: list[tuple[ProposedFact, GroundingDecision]] = []
        rejected: list[dict[str, Any]] = []
        for fact in proposed:
            decision = decisions[fact.temp_fact_id]
            if decision.accepted:
                accepted.append((fact, decision))
            else:
                rejected.append({"fact": fact.to_dict(), "decision": asdict(decision)})
        return accepted, rejected, [usage]

    def _complete(
        self, *, stage: str, role: str, prompt: str, max_new_tokens: int
    ) -> tuple[LLMResponse, StageUsage]:
        model = self.models[role]
        response = self.client.complete(
            model_spec=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            json_mode=True,
        )
        price = self.prices.get(model.model_name) or self.prices.get(model.effective_model_name)
        if price is None:
            raise ValueError(f"missing price for reference model: {model.model_name}")
        usage = StageUsage(
            stage=stage,
            role=role,
            configured_model=model.model_name,
            request_model=model.effective_model_name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            input_cost=response.input_tokens * price.official_price_in_per_1m / 1_000_000,
            output_cost=response.output_tokens * price.official_price_out_per_1m / 1_000_000,
            usage_source=response.usage_source,
            retry_count=response.retry_count,
            latency_ms=response.latency_ms,
            provider_request_id=response.provider_request_id,
            finish_reason=response.finish_reason,
        )
        return response, usage

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


def _source_annotated_text(segment: TopicSegment) -> str:
    """Expose canonical source IDs even when legacy text contains 0-based labels."""
    # A LongMemEval turn may contain paragraphs or example dialogue. Find only
    # the expected sequential legacy header (canonical turn_id - 1), preventing
    # numbered lists/example messages inside one turn from becoming boundaries.
    starts: list[int] = []
    cursor = 0
    for turn_id in segment.turn_ids:
        header = re.compile(
            rf"(?m)^\[[^\]\r\n]+\]\s+{int(turn_id) - 1}\.[^:\r\n]+:\s*"
        )
        match = header.search(segment.text, cursor)
        if match is None:
            raise ValueError(
                f"cannot locate source turn {turn_id} in segment text: {segment.segment_id}"
            )
        starts.append(match.start())
        cursor = match.end()
    if not starts or starts[0] != 0:
        raise ValueError(f"segment text has an unrecognized prefix: {segment.segment_id}")
    parts = [
        segment.text[start : starts[index + 1] if index + 1 < len(starts) else len(segment.text)]
        for index, start in enumerate(starts)
    ]
    return "\n".join(
        f"<SOURCE_TURN_ID={turn_id}> {part}"
        for turn_id, part in zip(segment.turn_ids, parts, strict=True)
    )

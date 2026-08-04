"""Crash-safe L/M/H candidate generation with repair, audit, and resume support."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from infobudget.rl_router.api import ModelAPIError
from infobudget.rl_router.buffers import (
    OversizeSegmentError,
    build_tier_buffers,
    tier_config_int,
)
from infobudget.rl_router.costs import allocate_batch
from infobudget.rl_router.embedding import Encoder
from infobudget.rl_router.ledger import SqliteLedger, atomic_write_json
from infobudget.rl_router.parsing import (
    is_schema_repairable,
    parse_fact_batch,
    render_extraction_prompt,
    render_json_repair_prompt,
)
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.run_state import ExtractionRunState, RunFileLock
from infobudget.rl_router.schemas import (
    BatchCompletion,
    FactRecord,
    ProviderUsage,
    Tier,
    TopicSegment,
)
from infobudget.schemas import ModelSpec, PriceSpec

CompletionFunction = Callable[[Tier, str, int], BatchCompletion]


class ProviderCircuitOpenError(RuntimeError):
    """Stop a tier after a non-retryable provider/configuration failure."""

    def __init__(self, tier: Tier, message: str):
        super().__init__(f"provider circuit opened for tier {tier}: {message}")
        self.tier = tier


def prepare_extraction_segments(
    *,
    segments: list[TopicSegment],
    prompts: dict[Tier, str],
    extraction_config: dict[str, Any],
    token_counters: dict[Tier, Callable[[str], int]],
    tiers: tuple[Tier, ...] | list[Tier],
) -> tuple[list[TopicSegment], dict[Tier, dict[str, dict[str, Any]]]]:
    """Tail-truncate segments that cannot fit a safe singleton request."""
    selected = tuple(tiers)
    prepared: list[TopicSegment] = []
    truncation_plan: dict[Tier, dict[str, dict[str, Any]]] = {
        tier: {} for tier in selected
    }
    for segment in segments:
        oversized_for = [
            tier
            for tier in selected
            if not _fits_total_context(
                segment, tier, prompts, extraction_config, token_counters
            )
        ]
        if not oversized_for:
            prepared.append(segment)
            continue
        if not bool(extraction_config.get("truncate_over_total_context", False)):
            raise OversizeSegmentError(
                f"oversize_segment: {segment.segment_id} tiers={oversized_for}"
            )
        truncated, audit = _truncate_segment_to_fit(
            segment,
            selected,
            prompts,
            extraction_config,
            token_counters,
        )
        prepared.append(truncated)
        for tier in selected:
            truncation_plan[tier][segment.segment_id] = {
                "strategy": audit["strategy"],
                "original_char_count": audit["original_char_count"],
                "retained_char_count": audit["retained_char_count"],
                "dropped_char_count": audit["dropped_char_count"],
                "original_input_tokens": audit["original_input_tokens"][tier],
                "truncated_input_tokens": audit["truncated_input_tokens"][tier],
                "visible_source_ids": audit["visible_source_ids"],
                "dropped_source_ids": audit["dropped_source_ids"],
            }
    return prepared, truncation_plan


def _fits_total_context(
    segment: TopicSegment,
    tier: Tier,
    prompts: dict[Tier, str],
    extraction_config: dict[str, Any],
    token_counters: dict[Tier, Callable[[str], int]],
) -> bool:
    input_tokens = token_counters[tier](
        render_extraction_prompt(prompts[tier], tier, [segment])
    )
    buffer_config = extraction_config["buffers"][tier]
    reserve = tier_config_int(extraction_config, "reserve_output_tokens_per_segment", tier)
    return input_tokens + reserve <= int(buffer_config["max_total_context_tokens"])


TRUNCATION_MARKER = "\n[TRUNCATED_TO_FIT_CONTEXT: trailing content omitted]"


def _truncate_segment_to_fit(
    segment: TopicSegment,
    tiers: tuple[Tier, ...],
    prompts: dict[Tier, str],
    extraction_config: dict[str, Any],
    token_counters: dict[Tier, Callable[[str], int]],
) -> tuple[TopicSegment, dict[str, Any]]:
    original = segment.text.rstrip()
    if not original:
        raise OversizeSegmentError(
            f"oversize_segment has no truncatable text: {segment.segment_id}"
        )

    original_input_tokens = {
        tier: token_counters[tier](
            render_extraction_prompt(prompts[tier], tier, [segment])
        )
        for tier in tiers
    }

    def candidate(retained_chars: int) -> TopicSegment:
        retained = original[:retained_chars].rstrip()
        text = f"{retained}{TRUNCATION_MARKER}" if retained else TRUNCATION_MARKER.strip()
        return replace(
            segment,
            text=text,
            token_count=max(token_counters[tier](text) for tier in tiers),
            extraction_truncated=True,
            extraction_original_char_count=len(original),
            extraction_retained_char_count=len(retained),
        )

    low, high, best = 1, len(original) - 1, None
    while low <= high:
        midpoint = (low + high) // 2
        proposed = candidate(midpoint)
        if all(
            _fits_total_context(
                proposed, tier, prompts, extraction_config, token_counters
            )
            for tier in tiers
        ):
            best = proposed
            low = midpoint + 1
        else:
            high = midpoint - 1
    if best is None:
        raise OversizeSegmentError(
            "segment cannot fit the total-context budget even after tail truncation: "
            f"{segment.segment_id}"
        )

    visible_source_ids = _visible_source_ids(best.text)
    if not visible_source_ids:
        raise OversizeSegmentError(
            "segment truncation removed every complete source line: "
            f"{segment.segment_id}"
        )
    best = replace(
        best,
        extraction_visible_source_ids=visible_source_ids,
    )

    truncated_input_tokens = {
        tier: token_counters[tier](
            render_extraction_prompt(prompts[tier], tier, [best])
        )
        for tier in tiers
    }
    return best, {
        "strategy": "tail_truncation",
        "original_char_count": len(original),
        "retained_char_count": best.extraction_retained_char_count,
        "dropped_char_count": len(original) - best.extraction_retained_char_count,
        "original_input_tokens": original_input_tokens,
        "truncated_input_tokens": truncated_input_tokens,
        "visible_source_ids": list(visible_source_ids),
        "dropped_source_ids": sorted(
            {turn_id - 1 for turn_id in segment.turn_ids} - set(visible_source_ids)
        ),
    }


def batch_output_token_limit(
    extraction_config: dict,
    tier: Tier,
    segment_count: int,
    model_spec: ModelSpec,
) -> int:
    """Reserve output for the actual batch without exceeding the provider model cap."""
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    value = min(
        tier_config_int(extraction_config, "reserve_output_tokens_per_segment", tier)
        * segment_count,
        model_spec.max_output_tokens,
    )
    if value <= 0:
        raise ValueError(f"model {tier} must configure max_output_tokens")
    return value


@dataclass(slots=True)
class CandidateGenerationSummary:
    sample_id: str
    extraction_run_id: str
    fact_counts: dict[Tier, int]
    batch_counts: dict[Tier, int]
    batch_status_counts: dict[str, int]
    batch_status_by_tier: dict[Tier, dict[str, int]]
    selected_tiers: list[Tier]
    status: str
    known_cost: float
    unknown_cost_attempts: int
    attempt_summary: dict[str, Any]
    quality_metrics: dict[str, Any]


def estimate_candidate_plan(
    *,
    segments: list[TopicSegment],
    prompts: dict[Tier, str],
    extraction_config: dict[str, Any],
    token_counters: dict[Tier, Callable[[str], int]],
    models: dict[Tier, ModelSpec],
    prices: dict[Tier, PriceSpec],
) -> dict[str, Any]:
    """Preflight deterministic batches and a conservative no-repair token budget."""
    result: dict[str, Any] = {
        tier: {"batch_count": 0, "input_tokens": 0, "reserved_output_tokens": 0}
        for tier in models
    }

    def planned(tier: Tier, _sample_id: str, batch: list[TopicSegment], prompt: str) -> None:
        result[tier]["batch_count"] += 1
        result[tier]["input_tokens"] += token_counters[tier](prompt)
        result[tier]["reserved_output_tokens"] += batch_output_token_limit(
            extraction_config, tier, len(batch), models[tier]
        )

    tiers = tuple(models)
    buffers = build_tier_buffers(
        prompts, extraction_config, token_counters, planned, tiers=tiers
    )
    for segment in segments:
        for tier in tiers:
            buffers[tier].add(segment)
    for buffer in buffers.values():
        buffer.finalize()
    for tier, values in result.items():
        price = prices[tier]
        values["maximum_cost_without_retries"] = (
            values["input_tokens"] * price.official_price_in_per_1m / 1_000_000
            + values["reserved_output_tokens"] * price.official_price_out_per_1m / 1_000_000
        )
        values["currency"] = price.currency
    return result


class CandidateGenerator:
    def __init__(
        self,
        *,
        store: FactQdrantStore,
        encoder: Encoder,
        models: dict[Tier, ModelSpec],
        prices: dict[Tier, PriceSpec],
        token_counters: dict[Tier, Callable[[str], int]],
        completion: CompletionFunction,
        prompts: dict[Tier, str],
        prompt_versions: dict[Tier, str],
        extraction_config: dict,
        output_root: str | Path,
    ):
        self.store, self.encoder, self.models, self.prices = store, encoder, models, prices
        self.token_counters, self.completion = token_counters, completion
        self.prompts, self.prompt_versions, self.cfg = prompts, prompt_versions, extraction_config
        self.output_root = Path(output_root)
        self._run_id = ""
        self._run_dir = Path()
        self._state: ExtractionRunState | None = None
        self._fact_counts = {tier: 0 for tier in models}
        self._batch_counts = {tier: 0 for tier in models}

    def generate(
        self,
        segments: list[TopicSegment],
        extraction_run_id: str | None = None,
        *,
        resume: bool = False,
        retry_terminal: bool = False,
        tiers: tuple[Tier, ...] | list[Tier] | None = None,
    ) -> CandidateGenerationSummary:
        if not segments:
            raise ValueError("candidate generation requires segments")
        sample_id = segments[0].sample_id
        if any(segment.sample_id != sample_id for segment in segments):
            raise ValueError("candidate generation cannot mix samples")
        selected_tiers = tuple(tiers or self.models)
        if not selected_tiers or len(set(selected_tiers)) != len(selected_tiers):
            raise ValueError("candidate generation tiers must be non-empty and unique")
        unknown_tiers = [tier for tier in selected_tiers if tier not in self.models]
        if unknown_tiers:
            raise ValueError(f"unknown candidate generation tiers: {unknown_tiers}")
        self._run_id = extraction_run_id or str(uuid.uuid4())
        self._run_dir = self.output_root / "runs" / self._run_id
        self._fact_counts = {tier: 0 for tier in self.models}
        self._batch_counts = {tier: 0 for tier in self.models}
        scope_hash = _scope_hash(segments, self.prompts, self.models, self.cfg)

        with RunFileLock(self._run_dir / "run.lock"):
            self._state = ExtractionRunState(self._run_dir, self._run_id)
            try:
                self._state.register_run(scope_hash, resume=resume)
                if retry_terminal:
                    if not resume:
                        raise ValueError("retry_terminal requires resume=True")
                    terminal_batch_ids = self._state.batch_ids(
                        "failed_terminal", selected_tiers
                    )
                    self._state.retry_terminal(selected_tiers)
                    for batch_id in terminal_batch_ids:
                        self._selected_response_path(batch_id).unlink(missing_ok=True)
                buffers = build_tier_buffers(
                    self.prompts,
                    self.cfg,
                    self.token_counters,
                    self._flush,
                    tiers=selected_tiers,
                )
                for segment in segments:
                    for tier in selected_tiers:
                        buffers[tier].add(segment)
                for buffer in buffers.values():
                    buffer.finalize()
                state_summary = self._state.summary()
                complete = all(
                    state_summary["by_tier"].get(tier)
                    and state_summary["by_tier"][tier].get("committed", 0)
                    == sum(state_summary["by_tier"][tier].values())
                    for tier in selected_tiers
                )
                run_status = "complete" if complete else "incomplete"
                self._state.finish_run(run_status)
                self._refresh_counts(segments[0])
                cost = self._attempt_cost_summary()
                quality = self._quality_metrics(segments[0], state_summary)
                return CandidateGenerationSummary(
                    sample_id,
                    self._run_id,
                    dict(self._fact_counts),
                    {
                        tier: sum(state_summary["by_tier"].get(tier, {}).values())
                        for tier in self.models
                    },
                    state_summary["by_status"],
                    {
                        tier: dict(state_summary["by_tier"].get(tier, {}))
                        for tier in self.models
                    },
                    list(selected_tiers),
                    run_status,
                    cost["known_cost"],
                    cost["unknown_cost_attempts"],
                    cost,
                    quality,
                )
            finally:
                self._state.close()
                self._state = None

    def _flush(
        self, tier: Tier, sample_id: str, segments: list[TopicSegment], prompt: str
    ) -> None:
        if self._state is None:
            raise RuntimeError("candidate run state is not initialized")
        sequence_index = self._batch_counts[tier]
        segment_ids = [segment.segment_id for segment in segments]
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        batch_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self._run_id}:{sample_id}:{tier}:{prompt_hash}:{json.dumps(segment_ids)}",
            )
        )
        batch_ledger, segment_ledger, failure_ledger = self._ledgers(segments[0], sample_id)
        try:
            status = self._state.plan_batch(
                batch_id=batch_id,
                sample_id=sample_id,
                tier=tier,
                sequence_index=sequence_index,
                segment_ids=segment_ids,
                prompt_hash=prompt_hash,
            )
            if status in {"committed", "failed_terminal"}:
                return
            max_new_tokens = batch_output_token_limit(
                self.cfg, tier, len(segments), self.models[tier]
            )
            response = self._load_selected_response(batch_id)
            if response is None:
                self._state.mark(batch_id, "requesting")
                response = self._call_model(
                    tier=tier,
                    batch_id=batch_id,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    call_type="initial",
                )
                if response is None:
                    self._state.mark(batch_id, "failed_retryable", "model request failed")
                    self._record_failure(
                        failure_ledger, batch_id, sample_id, tier, segment_ids,
                        "model request failed", "failed_retryable"
                    )
                    return
                self._state.mark(batch_id, "api_succeeded")
            if not self._provider_usage_allowed(response):
                error = "provider usage is required but missing"
                self._state.mark(batch_id, "failed_terminal", error)
                self._record_failure(
                    failure_ledger, batch_id, sample_id, tier, segment_ids,
                    error, "failed_terminal"
                )
                return
            if _response_was_truncated(response):
                error = f"model output was truncated: finish_reason={response.usage.finish_reason}"
                self._state.mark(batch_id, "failed_terminal", error)
                self._record_failure(
                    failure_ledger, batch_id, sample_id, tier, segment_ids,
                    error, "failed_terminal"
                )
                return

            expected_sources = {
                segment.segment_id: _allowed_source_ids(segment)
                for segment in segments
            }
            max_facts = tier_config_int(self.cfg, "max_facts_per_segment", tier)
            repair_limit = int(self.cfg.get("schema_repair_max_attempts", 2))
            repairs = 0
            while True:
                try:
                    parsed = parse_fact_batch(
                        response.content, segment_ids, max_facts, expected_sources
                    )
                    break
                except ValueError as exc:
                    if repairs >= repair_limit or not is_schema_repairable(exc):
                        self._state.mark(batch_id, "failed_terminal", str(exc))
                        self._record_failure(
                            failure_ledger, batch_id, sample_id, tier, segment_ids,
                            str(exc), "failed_terminal"
                        )
                        return
                    repairs += 1
                    repair_prompt = render_json_repair_prompt(
                        invalid_output=response.content,
                        validation_error=str(exc),
                        expected_segment_ids=segment_ids,
                        expected_source_ids_by_segment=expected_sources,
                        max_facts_per_segment=max_facts,
                    )
                    repair_input_tokens = self.token_counters[tier](repair_prompt)
                    repair_max_new_tokens = min(
                        max_new_tokens,
                        self.models[tier].max_context_tokens - repair_input_tokens,
                    )
                    if repair_max_new_tokens <= 0:
                        error = "schema repair prompt exceeds model context capacity"
                        self._state.mark(batch_id, "failed_terminal", error)
                        self._record_failure(
                            failure_ledger, batch_id, sample_id, tier, segment_ids,
                            error, "failed_terminal"
                        )
                        return
                    repaired = self._call_model(
                        tier=tier,
                        batch_id=batch_id,
                        prompt=repair_prompt,
                        max_new_tokens=repair_max_new_tokens,
                        call_type="schema_repair",
                    )
                    if repaired is None:
                        self._state.mark(
                            batch_id, "failed_retryable", "schema repair request failed"
                        )
                        self._record_failure(
                            failure_ledger, batch_id, sample_id, tier, segment_ids,
                            "schema repair request failed", "failed_retryable"
                        )
                        return
                    response = repaired
                    self._state.mark(batch_id, "api_succeeded")
                    if not self._provider_usage_allowed(response):
                        error = "provider usage is required but missing from schema repair"
                        self._state.mark(batch_id, "failed_terminal", error)
                        self._record_failure(
                            failure_ledger, batch_id, sample_id, tier, segment_ids,
                            error, "failed_terminal"
                        )
                        return
                    if _response_was_truncated(response):
                        error = (
                            "schema repair output was truncated: "
                            f"finish_reason={response.usage.finish_reason}"
                        )
                        self._state.mark(batch_id, "failed_terminal", error)
                        self._record_failure(
                            failure_ledger, batch_id, sample_id, tier, segment_ids,
                            error, "failed_terminal"
                        )
                        return

            self._state.mark(batch_id, "parsed")
            usage = self._aggregate_batch_usage(batch_id, tier)
            input_weights = [self.token_counters[tier](segment.text) for segment in segments]
            output_weights = [
                self.token_counters[tier](parsed.block_text_by_segment[segment.segment_id])
                for segment in segments
            ]
            allocations = allocate_batch(
                usage,
                segment_ids,
                input_weights,
                output_weights,
                [len(parsed.facts_by_segment[segment.segment_id]) for segment in segments],
                self.prices[tier],
            )
            facts: list[FactRecord] = []
            created_at = datetime.now(timezone.utc).isoformat()
            for segment, allocation in zip(segments, allocations):
                fact_texts = parsed.facts_by_segment[segment.segment_id]
                source_id_groups = parsed.source_ids_by_segment[segment.segment_id]
                for fact_index, (source_ids, text) in enumerate(
                    zip(source_id_groups, fact_texts)
                ):
                    fact_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{self._run_id}:{tier}:{segment.segment_id}:{fact_index}:{text}",
                        )
                    )
                    count = len(fact_texts)
                    facts.append(
                        FactRecord(
                            fact_id=fact_id,
                            dataset_name=segment.dataset_name,
                            split=segment.split,
                            sample_id=sample_id,
                            session_id=segment.session_id,
                            segment_id=segment.segment_id,
                            segment_hash=segment.source_content_hash,
                            source_turn_ids=[source_id + 1 for source_id in source_ids],
                            fact_text=text,
                            fact_index=fact_index,
                            fact_count_in_segment=count,
                            memory_tier=tier,
                            extractor_model=self.models[tier].model_name,
                            prompt_version=self.prompt_versions[tier],
                            batch_id=batch_id,
                            extraction_run_id=self._run_id,
                            segment_start_timestamp=segment.start_timestamp,
                            segment_end_timestamp=segment.end_timestamp,
                            allocated_input_tokens=allocation.input_tokens / count,
                            allocated_output_tokens=allocation.output_tokens / count,
                            allocated_total_tokens=allocation.total_tokens / count,
                            allocated_input_cost=allocation.input_cost / count,
                            allocated_output_cost=allocation.output_cost / count,
                            allocated_total_cost=allocation.total_cost / count,
                            embedding_model=self.encoder.model_name,
                            embedding_dimension=self.encoder.dimension,
                            segment_order=segment.segment_order,
                            created_at=created_at,
                            extra={
                                **_source_provenance(segment, source_ids),
                                **_extraction_truncation_metadata(segment),
                            },
                        )
                    )
            vectors = (
                self.encoder.encode([fact.fact_text for fact in facts])
                if facts
                else []
            )
            self.store.replace_candidate_batch(
                tier,
                dataset_name=segments[0].dataset_name,
                split=segments[0].split,
                sample_id=sample_id,
                extraction_run_id=self._run_id,
                batch_id=batch_id,
                facts=facts,
                vectors=vectors,
            )
            stored_count = sum(
                len(
                    self.store.candidate_points(
                        tier,
                        dataset_name=segments[0].dataset_name,
                        split=segments[0].split,
                        sample_id=sample_id,
                        segment_id=segment.segment_id,
                        extraction_run_id=self._run_id,
                        batch_id=batch_id,
                        with_vectors=False,
                    )
                )
                for segment in segments
            )
            if stored_count != len(facts):
                raise RuntimeError(
                    f"Qdrant audit mismatch for {batch_id}: expected {len(facts)}, got {stored_count}"
                )
            self._state.mark(batch_id, "embedded")
            for segment, allocation in zip(segments, allocations):
                fact_texts = parsed.facts_by_segment[segment.segment_id]
                segment_ledger.upsert(
                    {
                        "extraction_run_id": self._run_id,
                        "batch_id": batch_id,
                        "segment_id": segment.segment_id,
                        "extraction_truncated": segment.extraction_truncated,
                        "extraction_original_char_count": segment.extraction_original_char_count,
                        "extraction_retained_char_count": segment.extraction_retained_char_count,
                        "extraction_visible_source_ids": list(
                            segment.extraction_visible_source_ids
                        ),
                        "extraction_dropped_source_ids": _dropped_source_ids(segment),
                        "tier": tier,
                        "allocated_input_tokens": allocation.input_tokens,
                        "allocated_output_tokens": allocation.output_tokens,
                        "allocated_total_tokens": allocation.total_tokens,
                        "allocated_input_cost": allocation.input_cost,
                        "allocated_output_cost": allocation.output_cost,
                        "allocated_total_cost": allocation.total_cost,
                        "serialized_input_tokens": allocation.serialized_input_tokens,
                        "attributed_output_tokens": allocation.attributed_output_tokens,
                        "fact_count": len(fact_texts),
                        "status": "ok" if fact_texts else "no_fact",
                    }
                )
            price = self.prices[tier]
            attempts = self._batch_attempt_rows(batch_id)
            batch_ledger.upsert(
                {
                    "extraction_run_id": self._run_id,
                    "batch_id": batch_id,
                    "sample_id": sample_id,
                    "tier": tier,
                    "model_name": usage.model_name,
                    "segment_ids": segment_ids,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "usage_source": usage.usage_source,
                    "input_cost": usage.input_tokens * price.official_price_in_per_1m / 1_000_000,
                    "output_cost": usage.output_tokens * price.official_price_out_per_1m / 1_000_000,
                    "input_price_per_1m": price.official_price_in_per_1m,
                    "output_price_per_1m": price.official_price_out_per_1m,
                    "price_effective_date": price.price_effective_date,
                    "currency": price.currency,
                    "logical_call_count": len({row["logical_call_index"] for row in attempts}),
                    "unknown_cost_attempts": sum(row["cost_status"] == "unknown" for row in attempts),
                    "status": "ok",
                    "created_at": created_at,
                }
            )
            self._state.mark(batch_id, "committed")
        except ProviderCircuitOpenError as exc:
            if self._state.status(batch_id) not in {"failed_terminal", "committed"}:
                self._state.mark(batch_id, "failed_terminal", str(exc))
            self._record_failure(
                failure_ledger, batch_id, sample_id, tier, segment_ids,
                str(exc), self._state.status(batch_id)
            )
            raise
        except Exception as exc:
            if self._state.status(batch_id) not in {"failed_terminal", "committed"}:
                self._state.mark(batch_id, "failed_retryable", str(exc))
            self._record_failure(
                failure_ledger, batch_id, sample_id, tier, segment_ids,
                str(exc), self._state.status(batch_id)
            )
        finally:
            self._batch_counts[tier] += 1

    def _call_model(
        self,
        *,
        tier: Tier,
        batch_id: str,
        prompt: str,
        max_new_tokens: int,
        call_type: str,
    ) -> BatchCompletion | None:
        call_index = self._next_logical_call_index(batch_id)
        call_dir = self._run_dir / "raw" / tier / batch_id / f"call_{call_index:03d}_{call_type}"
        atomic_write_json(
            call_dir / "request.json",
            {
                "extraction_run_id": self._run_id,
                "batch_id": batch_id,
                "logical_call_index": call_index,
                "call_type": call_type,
                "model": self.models[tier].effective_model_name,
                "max_new_tokens": max_new_tokens,
                "json_mode": True,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
            },
        )
        try:
            response = self.completion(tier, prompt, max_new_tokens)
        except Exception as exc:
            attempts = exc.attempts if isinstance(exc, ModelAPIError) else []
            if not attempts:
                attempts = [
                    {
                        "transport_attempt": 1,
                        "status": "failed",
                        "http_status": None,
                        "latency_ms": 0,
                        "provider_request_id": "",
                        "error": str(exc)[:500],
                        "cost_status": "unknown",
                    }
                ]
            self._record_attempts(
                tier, batch_id, call_index, call_type, prompt, None, attempts
            )
            atomic_write_json(
                call_dir / "error.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": getattr(exc, "retryable", True),
                    "attempts": attempts,
                },
            )
            if isinstance(exc, ModelAPIError) and not exc.retryable:
                raise ProviderCircuitOpenError(tier, str(exc)) from exc
            return None
        attempts = response.attempts or [
            {
                "transport_attempt": 1,
                "status": "succeeded",
                "http_status": 200,
                "latency_ms": response.usage.latency_ms,
                "provider_request_id": "",
                "cost_status": "reported_on_logical_response",
            }
        ]
        self._record_attempts(
            tier, batch_id, call_index, call_type, prompt, response, attempts
        )
        atomic_write_json(
            call_dir / "response.json",
            {
                "content": response.content,
                "usage": asdict(response.usage),
                "attempts": attempts,
            },
        )
        atomic_write_json(
            self._selected_response_path(batch_id),
            {"content": response.content, "usage": asdict(response.usage)},
        )
        return response

    def _provider_usage_allowed(self, response: BatchCompletion) -> bool:
        return (
            not self.cfg.get("require_provider_usage", True)
            or response.usage.usage_source == "provider"
        )

    def _record_attempts(
        self,
        tier: Tier,
        batch_id: str,
        call_index: int,
        call_type: str,
        prompt: str,
        response: BatchCompletion | None,
        attempts: list[dict[str, Any]],
    ) -> None:
        ledger = self._attempt_ledger()
        price = self.prices[tier]
        estimated_input = self.token_counters[tier](prompt)
        last_success = max(
            (int(item.get("transport_attempt", 1)) for item in attempts if item.get("status") == "succeeded"),
            default=-1,
        )
        for item in attempts:
            transport_attempt = int(item.get("transport_attempt", 1))
            succeeded = response is not None and transport_attempt == last_success
            input_tokens = response.usage.input_tokens if succeeded else None
            output_tokens = response.usage.output_tokens if succeeded else None
            usage_source = response.usage.usage_source if succeeded else "unavailable"
            if succeeded:
                cost_status = "known" if usage_source == "provider" else "estimated"
                input_cost = input_tokens * price.official_price_in_per_1m / 1_000_000
                output_cost = output_tokens * price.official_price_out_per_1m / 1_000_000
            else:
                cost_status, input_cost, output_cost = "unknown", None, None
            ledger.append(
                {
                    "extraction_run_id": self._run_id,
                    "batch_id": batch_id,
                    "logical_call_index": call_index,
                    "transport_attempt": transport_attempt,
                    "call_type": call_type,
                    "tier": tier,
                    "model_name": self.models[tier].model_name,
                    "status": item.get("status", "failed"),
                    "http_status": item.get("http_status"),
                    "latency_ms": int(item.get("latency_ms") or 0),
                    "provider_request_id": str(item.get("provider_request_id") or ""),
                    "started_at": item.get("started_at"),
                    "finish_reason": response.usage.finish_reason if succeeded else "",
                    "usage_source": usage_source,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_input_tokens_if_billed": estimated_input,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "cost_status": cost_status,
                    "currency": price.currency,
                    "error": item.get("error"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def _aggregate_batch_usage(self, batch_id: str, tier: Tier) -> ProviderUsage:
        rows = [
            row
            for row in self._batch_attempt_rows(batch_id)
            if row["status"] == "succeeded" and row["input_tokens"] is not None
        ]
        if not rows:
            raise RuntimeError(f"no auditable successful usage for batch {batch_id}")
        sources = {str(row["usage_source"]) for row in rows}
        return ProviderUsage(
            input_tokens=sum(int(row["input_tokens"]) for row in rows),
            output_tokens=sum(int(row["output_tokens"]) for row in rows),
            model_name=self.models[tier].model_name,
            usage_source="provider" if sources == {"provider"} else "mixed_estimate",
            retry_count=sum(row["status"] == "failed" for row in self._batch_attempt_rows(batch_id)),
            latency_ms=sum(int(row["latency_ms"]) for row in self._batch_attempt_rows(batch_id)),
            provider_request_id=str(rows[-1].get("provider_request_id") or ""),
            finish_reason=str(rows[-1].get("finish_reason") or ""),
        )

    def _load_selected_response(self, batch_id: str) -> BatchCompletion | None:
        path = self._selected_response_path(batch_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return BatchCompletion(payload["content"], ProviderUsage(**payload["usage"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _selected_response_path(self, batch_id: str) -> Path:
        return self._run_dir / "raw" / "selected" / f"{batch_id}.json"

    def _attempt_ledger(self) -> SqliteLedger:
        return SqliteLedger(
            self._run_dir / "ledgers" / "run_ledger.sqlite3",
            "attempts",
            ("extraction_run_id", "batch_id", "logical_call_index", "transport_attempt"),
            legacy_jsonl_path=self._run_dir / "ledgers" / "attempts.jsonl",
        )

    def _batch_attempt_rows(self, batch_id: str) -> list[dict[str, Any]]:
        return [
            row for row in self._attempt_ledger().read_all()
            if row.get("extraction_run_id") == self._run_id and row.get("batch_id") == batch_id
        ]

    def _next_logical_call_index(self, batch_id: str) -> int:
        return 1 + max(
            (int(row["logical_call_index"]) for row in self._batch_attempt_rows(batch_id)),
            default=0,
        )

    def _attempt_cost_summary(self) -> dict[str, Any]:
        rows = self._attempt_ledger().read_all()
        return {
            "known_cost": sum(
                float(row.get("input_cost") or 0.0) + float(row.get("output_cost") or 0.0)
                for row in rows
                if row.get("cost_status") in {"known", "estimated"}
            ),
            "unknown_cost_attempts": sum(row.get("cost_status") == "unknown" for row in rows),
            "successful_attempts": sum(row.get("status") == "succeeded" for row in rows),
            "failed_attempts": sum(row.get("status") == "failed" for row in rows),
            "repair_calls": len(
                {
                    (row.get("batch_id"), row.get("logical_call_index"))
                    for row in rows if row.get("call_type") == "schema_repair"
                }
            ),
        }

    def _refresh_counts(self, segment: TopicSegment) -> None:
        _, segment_ledger, _ = self._ledgers(segment, segment.sample_id)
        rows = [
            row for row in segment_ledger.read_all()
            if row.get("extraction_run_id") == self._run_id
        ]
        self._fact_counts = {
            tier: sum(int(row.get("fact_count", 0)) for row in rows if row.get("tier") == tier)
            for tier in self.models
        }

    def _quality_metrics(
        self, segment: TopicSegment, state_summary: dict[str, Any]
    ) -> dict[str, Any]:
        _, segment_ledger, _ = self._ledgers(segment, segment.sample_id)
        rows = [
            row
            for row in segment_ledger.read_all()
            if row.get("extraction_run_id") == self._run_id
        ]
        total_segment_results = len(rows)
        empty_segments = sum(row.get("status") == "no_fact" for row in rows)
        saturated_segments = sum(
            int(row.get("fact_count", 0))
            >= tier_config_int(self.cfg, "max_facts_per_segment", row["tier"])
            for row in rows
        )
        attempts = [
            row
            for row in self._attempt_ledger().read_all()
            if row.get("extraction_run_id") == self._run_id
        ]
        repair_batches = len(
            {
                row.get("batch_id")
                for row in attempts
                if row.get("call_type") == "schema_repair"
            }
        )
        total_batches = int(state_summary.get("batch_count", 0))
        failed_batches = sum(
            count
            for status, count in state_summary.get("by_status", {}).items()
            if status != "committed"
        )

        def rate(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        values: dict[str, Any] = {
            "total_segment_results": total_segment_results,
            "empty_fact_segments": empty_segments,
            "saturated_segments": saturated_segments,
            "total_batches": total_batches,
            "repair_batches": repair_batches,
            "failed_batches": failed_batches,
            "empty_fact_segment_rate": rate(empty_segments, total_segment_results),
            "saturated_segment_rate": rate(saturated_segments, total_segment_results),
            "repair_batch_rate": rate(repair_batches, total_batches),
            "failed_batch_rate": rate(failed_batches, total_batches),
        }
        gates = self.cfg.get("quality_gates", {})
        comparisons = {
            "empty_fact_segment_rate": float(
                gates.get("max_empty_fact_segment_rate", 1.0)
            ),
            "saturated_segment_rate": float(
                gates.get("max_saturated_segment_rate", 1.0)
            ),
            "repair_batch_rate": float(gates.get("max_repair_batch_rate", 1.0)),
            "failed_batch_rate": float(gates.get("max_failed_batch_rate", 0.0)),
        }
        violations = {
            key: {"actual": values[key], "maximum": maximum}
            for key, maximum in comparisons.items()
            if values[key] > maximum
        }
        values["thresholds"] = comparisons
        values["violations"] = violations
        values["passed"] = not violations
        return values

    def _record_failure(
        self,
        ledger: SqliteLedger,
        batch_id: str,
        sample_id: str,
        tier: Tier,
        segment_ids: list[str],
        error: str,
        status: str,
    ) -> None:
        existing = [
            row for row in ledger.read_all()
            if row.get("extraction_run_id") == self._run_id and row.get("batch_id") == batch_id
        ]
        ledger.append(
            {
                "extraction_run_id": self._run_id,
                "batch_id": batch_id,
                "failure_index": len(existing) + 1,
                "sample_id": sample_id,
                "tier": tier,
                "segment_ids": segment_ids,
                "error": error[:2000],
                "status": status,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _ledgers(self, segment: TopicSegment, sample_id: str):
        base = (
            self.output_root / segment.dataset_name / segment.split /
            segment.segmentation_method / "samples" / sample_id / "extraction"
        )
        return (
            SqliteLedger(
                base / "candidate_ledger.sqlite3",
                "batches",
                ("extraction_run_id", "batch_id"),
                legacy_jsonl_path=base / "batches.jsonl",
            ),
            SqliteLedger(
                base / "candidate_ledger.sqlite3",
                "segment_costs",
                ("extraction_run_id", "batch_id", "segment_id"),
                legacy_jsonl_path=base / "segment_costs.jsonl",
            ),
            SqliteLedger(
                base / "candidate_ledger.sqlite3",
                "failures",
                ("extraction_run_id", "batch_id", "failure_index"),
                legacy_jsonl_path=base / "failures.jsonl",
            ),
        )


def _scope_hash(
    segments: list[TopicSegment],
    prompts: dict[Tier, str],
    models: dict[Tier, ModelSpec],
    config: dict[str, Any],
) -> str:
    payload = {
        "dataset": segments[0].dataset_name,
        "split": segments[0].split,
        "sample_id": segments[0].sample_id,
        "segmentation_method": segments[0].segmentation_method,
        "segments": [
            [segment.segment_id, segment.source_content_hash] for segment in segments
        ],
        "prompt_hashes": {
            tier: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for tier, prompt in prompts.items()
        },
        "models": {tier: models[tier].effective_model_name for tier in models},
        "extraction_config": config,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


SOURCE_LINE_PATTERN = re.compile(
    r"^\[(?P<timestamp>[^,\]]+)(?:,\s*(?P<weekday>[^\]]+))?\]\s+"
    r"(?P<source_id>\d+)\.(?P<speaker>[^:]+):"
)


def _visible_source_ids(text: str) -> tuple[int, ...]:
    result: list[int] = []
    for line in text.splitlines():
        match = SOURCE_LINE_PATTERN.match(line.strip())
        if match is not None:
            source_id = int(match.group("source_id"))
            if source_id not in result:
                result.append(source_id)
    return tuple(result)


def _allowed_source_ids(segment: TopicSegment) -> set[int]:
    if segment.extraction_truncated:
        return set(segment.extraction_visible_source_ids)
    return {turn_id - 1 for turn_id in segment.turn_ids}


def _dropped_source_ids(segment: TopicSegment) -> list[int]:
    if not segment.extraction_truncated:
        return []
    return sorted(
        {turn_id - 1 for turn_id in segment.turn_ids}
        - set(segment.extraction_visible_source_ids)
    )


def _source_metadata(segment: TopicSegment, source_id: int) -> dict[str, str]:
    """Recover deterministic speaker/time fields from the frozen rendered source line."""
    for line in segment.text.splitlines():
        match = SOURCE_LINE_PATTERN.match(line.strip())
        if match is None or int(match.group("source_id")) != source_id:
            continue
        return {
            "source_speaker": match.group("speaker").strip(),
            "source_timestamp": match.group("timestamp").strip(),
            "source_weekday": (match.group("weekday") or "").strip(),
        }
    return {}


def _source_provenance(segment: TopicSegment, source_ids: list[int]) -> dict[str, Any]:
    """Build multi-source provenance while retaining singular reader compatibility fields."""
    records: list[dict[str, Any]] = []
    for source_id in source_ids:
        metadata = _source_metadata(segment, source_id)
        records.append(
            {
                "source_id": source_id,
                "source_turn_id": source_id + 1,
                "source_speaker": metadata.get("source_speaker", ""),
                "source_timestamp": metadata.get("source_timestamp", ""),
                "source_weekday": metadata.get("source_weekday", ""),
            }
        )
    result: dict[str, Any] = {
        "source_ids": list(source_ids),
        "source_provenance": records,
    }
    if records:
        primary = records[0]
        result.update(
            {
                "source_id": primary["source_id"],
                "source_speaker": primary["source_speaker"],
                "source_timestamp": primary["source_timestamp"],
                "source_weekday": primary["source_weekday"],
            }
        )
    return result


def _extraction_truncation_metadata(segment: TopicSegment) -> dict[str, Any]:
    if not segment.extraction_truncated:
        return {}
    return {
        "extraction_truncated": True,
        "extraction_original_char_count": segment.extraction_original_char_count,
        "extraction_retained_char_count": segment.extraction_retained_char_count,
        "extraction_dropped_char_count": (
            segment.extraction_original_char_count
            - segment.extraction_retained_char_count
        ),
        "extraction_visible_source_ids": list(
            segment.extraction_visible_source_ids
        ),
        "extraction_dropped_source_ids": _dropped_source_ids(segment),
    }


def _response_was_truncated(response: BatchCompletion) -> bool:
    return response.usage.finish_reason.strip().lower() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }

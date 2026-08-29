"""Core tests for the fact-only RL router."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
import uuid
from types import SimpleNamespace

from qdrant_client import models as qdrant_models

from infobudget.rl_router.buffers import (
    ExtractionBuffer,
    OversizeSegmentError,
    build_tier_buffers,
    tier_config_int,
)
from infobudget.rl_router.api import (
    LLMResponse,
    ModelAPIError,
    parse_chat_completion_response,
    require_api_keys,
)
from infobudget.rl_router.candidates import (
    CandidateGenerationSummary,
    CandidateGenerator,
    ProviderCircuitOpenError,
    _allowed_source_ids,
    _extraction_truncation_metadata,
    _source_metadata,
    _source_provenance,
    batch_output_token_limit,
    estimate_candidate_plan,
    estimate_routed_plan,
    prepare_extraction_segments,
    prepare_routed_extraction_segments,
)
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.costs import allocate_batch, normalize_virtual_cost, replay_virtual_cost
from infobudget.rl_router.deployment import (
    build_question_outcomes,
    summarize_deployment_costs,
    summarize_qa_usage,
    summarize_question_outcomes,
    validate_route_decisions,
)
from infobudget.rl_router.evaluation import (
    LightMemLLMJudge,
    LightMemQAReader,
    parse_judge_label,
    parse_locomo_judge_label,
    parse_longmemeval_judge_label,
    select_judge_prompt,
)
from infobudget.rl_router.export import export_memories
from infobudget.rl_router.experiment_identity import epoch_artifact_name
from infobudget.rl_router.ledger import SqliteLedger, atomic_write_json, read_sqlite_ledger
from infobudget.rl_router.metrics import summarize_fold_accuracy
from infobudget.rl_router.parsing import (
    parse_fact_batch,
    render_extraction_prompt,
    render_json_repair_prompt,
)
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.reconciliation import reconcile_extraction_run
from infobudget.rl_router.router import EmbeddingMLPRouter, FeatureScaler
from infobudget.rl_router.run_state import ExtractionRunState
from infobudget.rl_router.schemas import (
    BatchCompletion,
    FactRecord,
    ProviderUsage,
    ReplaySegmentCost,
    TopicSegment,
)
from infobudget.rl_router.training_io import load_replay_history
from infobudget.schemas import ModelSpec, PriceSpec

TIER_ID = {"small": 1, "medium": 2, "large": 3}


def test_epoch_artifact_name_is_stable_and_rejects_invalid_values() -> None:
    assert epoch_artifact_name(1) == "epochs_1"
    assert epoch_artifact_name(10) == "epochs_10"
    with pytest.raises(ValueError, match="epochs must be positive"):
        epoch_artifact_name(0)


def test_fold_accuracy_summary_reports_sample_stability() -> None:
    summary = summarize_fold_accuracy([0.6, 0.8, 1.0])
    assert summary["mean_fold_accuracy_micro"] == pytest.approx(0.8)
    assert summary["std_fold_accuracy_micro"] == pytest.approx(0.2)
    assert summary["sem_fold_accuracy_micro"] == pytest.approx(0.2 / np.sqrt(3))
    assert summary["min_fold_accuracy_micro"] == pytest.approx(0.6)
    assert summary["max_fold_accuracy_micro"] == pytest.approx(1.0)


def _sqlite_ledger_process_worker(path: str, worker: int) -> None:
    ledger = SqliteLedger(path, "rows", ("row_id",))
    for index in range(10):
        ledger.append({"row_id": f"{worker}:{index}", "worker": worker})
    ledger.append({"row_id": "shared", "worker": worker})


def test_qdrant_server_config_uses_remote_client_and_payload_indexes(monkeypatch) -> None:
    clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.indexes = []
            self.closed = False
            clients.append(self)

        def get_collections(self):
            return []

        def collection_exists(self, _name):
            return True

        def get_collection(self, _name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=qdrant_models.VectorParams(
                            size=1024, distance=qdrant_models.Distance.COSINE
                        )
                    )
                )
            )

        def create_payload_index(self, name, *, field_name, field_schema, wait):
            self.indexes.append((name, field_name, field_schema, wait))

        def close(self):
            self.closed = True

    monkeypatch.setattr("infobudget.rl_router.qdrant_store.QdrantClient", FakeClient)
    storage = {
        "mode": "server",
        "url": "http://127.0.0.1:6333",
        "grpc_port": 6334,
        "prefer_grpc": False,
        "timeout_seconds": 12,
        "api_key_env": "",
        "vector_size": 1024,
    }
    FactQdrantStore.probe_storage_config(storage)
    assert clients[0].closed is True
    store = FactQdrantStore.from_storage_config(
        storage,
        project_root=".",
        namespace="locomo_full_nsp_fact_v2",
    )
    assert store.mode == "server"
    assert clients[1].kwargs == {
        "url": "http://127.0.0.1:6333",
        "api_key": None,
        "timeout": 12.0,
        "prefer_grpc": False,
        "grpc_port": 6334,
    }
    assert len(clients[1].indexes) == 4 * 12
    assert all(wait is True for *_, wait in clients[1].indexes)
    store.close()


def test_qdrant_server_api_key_fails_before_client_creation(monkeypatch) -> None:
    monkeypatch.delenv("INFOBUDGET_QDRANT_TEST_KEY", raising=False)
    storage = {
        "mode": "server",
        "url": "http://127.0.0.1:6333",
        "api_key_env": "INFOBUDGET_QDRANT_TEST_KEY",
        "vector_size": 1024,
    }
    with pytest.raises(EnvironmentError, match="INFOBUDGET_QDRANT_TEST_KEY"):
        FactQdrantStore.probe_storage_config(storage)


def test_sqlite_ledger_is_cross_process_idempotent(tmp_path) -> None:
    path = tmp_path / "shared.sqlite3"
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_sqlite_ledger_process_worker, str(path), worker)
            for worker in range(2)
        ]
        for future in futures:
            future.result(timeout=30)
    rows = read_sqlite_ledger(path, "rows")
    assert len(rows) == 21
    assert sum(row["row_id"] == "shared" for row in rows) == 1


def _segment(index: int = 1, sample_id: str = "sample-1") -> TopicSegment:
    return TopicSegment(
        "locomo", "train", sample_id, "session_1",
        f"{sample_id}:nsp_text_tiling:seg_{index:06d}", "nsp_text_tiling", "nsp_text_tiling_v1",
        index, index, (index,), "2026-01-01T00:00:00.000", "2026-01-01T00:00:00.000",
        f"Alice: durable fact {index}", 5, f"hash-{index}", index,
    )


def _fact(source: TopicSegment, tier: str, run_id: str = "run") -> FactRecord:
    return FactRecord(
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source.segment_id}:{tier}:{run_id}")), source.dataset_name, source.split,
        source.sample_id, source.session_id, source.segment_id, source.source_content_hash,
        list(source.turn_ids), f"fact from {tier}", 0, 1, tier, f"model-{tier}", "v1", "batch", run_id,
        source.start_timestamp, source.end_timestamp, 10, 2, 12, .001, .001, .002,
        "fake", 4, source.segment_order,
    )


def test_fact_parser_no_fact_dedup_and_unknown_rejection() -> None:
    text = '''{
      "processed_segment_ids": ["s1", "s2"],
      "data": [
        {"segment_id": "s1", "source_id": 0, "fact": "Alice moved to Paris."},
        {"segment_id": "s1", "source_id": 0, "fact": "Alice moved to Paris."}
      ]
    }'''
    parsed = parse_fact_batch(text, ["s1", "s2"], 8, {"s1": {0}, "s2": {1}})
    assert parsed.facts_by_segment == {"s1": ["Alice moved to Paris."], "s2": []}
    assert parsed.source_ids_by_segment == {"s1": [[0]], "s2": []}
    with pytest.raises(ValueError, match="unknown segment_id"):
        parse_fact_batch(
            text.replace('"segment_id": "s1"', '"segment_id": "alien"'),
            ["s1", "s2"],
            8,
        )
    over_limit = text.replace(
        ']\n    }',
        ', {"segment_id": "s1", "source_id": 0, "fact": "Alice teaches."}]\n    }',
    )
    with pytest.raises(ValueError, match="exceeds max_facts_per_segment=1"):
        parse_fact_batch(over_limit, ["s1", "s2"], 1)


def test_fact_parser_supports_multiple_sources_and_merges_duplicate_provenance() -> None:
    parsed = parse_fact_batch(
        json.dumps(
            {
                "processed_segment_ids": ["s1"],
                "data": [
                    {
                        "segment_id": "s1",
                        "source_ids": [2, 0],
                        "fact": "Alice and Bob planned the trip.",
                    },
                    {
                        "segment_id": "s1",
                        "source_ids": [1, 2],
                        "fact": "Alice and Bob planned the trip.",
                    },
                ],
            }
        ),
        ["s1"],
        10,
        {"s1": {0, 1, 2}},
    )
    assert parsed.facts_by_segment == {"s1": ["Alice and Bob planned the trip."]}
    assert parsed.source_ids_by_segment == {"s1": [[0, 1, 2]]}
    assert '"source_ids":[0,1,2]' in parsed.block_text_by_segment["s1"]


def test_json_repair_prompt_contains_exact_contract_without_conversation_rewrite() -> None:
    prompt = render_json_repair_prompt(
        invalid_output='{"data": [}',
        validation_error="not valid JSON",
        expected_segment_ids=["s1", "s2"],
        expected_source_ids_by_segment={"s1": {0, 1}, "s2": {4}},
        max_facts_per_segment=10,
    )
    assert "Repair JSON structure only" in prompt
    assert '"processed_segment_ids": ["s1", "s2"]' in prompt
    assert '"s2": [4]' in prompt
    assert "no more than 10 facts for each topic segment" in prompt


def test_tier_buffers_use_distinct_prompts_and_output_limits() -> None:
    tiers = ("small", "medium", "large")
    prompts = {tier: f"prompt-{tier}" for tier in tiers}
    config = {
        "max_facts_per_segment": {"small": 4, "medium": 10, "large": 20},
        "reserve_output_tokens_per_segment": {"small": 64, "medium": 128, "large": 256},
        "buffers": {
            tier: {"max_segments": 8, "max_input_tokens": 1000, "max_total_context_tokens": 2000}
            for tier in tiers
        },
    }
    buffers = build_tier_buffers(
        prompts,
        config,
        {tier: lambda value: len(value.split()) for tier in tiers},
        lambda *_: None,
    )
    assert {tier: buffers[tier].prompt for tier in tiers} == prompts
    assert [buffers[tier].reserve_output_tokens_per_segment for tier in tiers] == [64, 128, 256]
    assert [tier_config_int(config, "max_facts_per_segment", tier) for tier in tiers] == [4, 10, 20]
    selected = build_tier_buffers(
        prompts,
        config,
        {"medium": lambda value: len(value.split())},
        lambda *_: None,
        tiers=("medium",),
    )
    assert tuple(selected) == ("medium",)


def test_batch_output_limit_uses_actual_segment_count_and_model_cap() -> None:
    config = {"reserve_output_tokens_per_segment": 1024}
    model = ModelSpec("api", "openai_compatible", "small", "small", 32768, 1, "n/a", 8192)
    assert batch_output_token_limit(config, "small", 1, model) == 1024
    assert batch_output_token_limit(config, "small", 8, model) == 8192
    assert batch_output_token_limit(config, "small", 10, model) == 8192


def test_fact_source_metadata_is_recovered_from_the_frozen_source_line() -> None:
    segment = replace(
        _segment(),
        text="[2023-05-08T13:56:00.000, Mon] 0.Alice: I visited Paris.",
    )
    assert _source_metadata(segment, 0) == {
        "source_speaker": "Alice",
        "source_timestamp": "2023-05-08T13:56:00.000",
        "source_weekday": "Mon",
    }


def test_fact_source_provenance_preserves_all_sources_and_primary_compatibility() -> None:
    segment = replace(
        _segment(),
        turn_ids=(1, 2),
        text=(
            "[2023-05-08T13:56:00.000, Mon] 0.Alice: I proposed Paris.\n"
            "[2023-05-08T13:57:00.000, Mon] 1.Bob: I agreed to Paris."
        ),
    )
    provenance = _source_provenance(segment, [0, 1])
    assert provenance["source_ids"] == [0, 1]
    assert [item["source_speaker"] for item in provenance["source_provenance"]] == [
        "Alice",
        "Bob",
    ]
    assert provenance["source_id"] == 0
    assert provenance["source_speaker"] == "Alice"


def test_api_key_preflight_checks_only_selected_tiers(monkeypatch) -> None:
    models = {
        tier: ModelSpec(
            "api",
            "openai_compatible",
            tier,
            tier,
            4096,
            1,
            "n/a",
            1024,
            api_key_env=f"TEST_{tier.upper()}_KEY",
        )
        for tier in ("small", "medium", "large")
    }
    monkeypatch.setenv("TEST_SMALL_KEY", "present")
    require_api_keys(models, ["small"], operation="test extraction")
    with pytest.raises(RuntimeError, match="medium=TEST_MEDIUM_KEY"):
        require_api_keys(models, ["small", "medium"], operation="test extraction")


def test_legacy_prompt_placeholders_render_without_formatting_json_braces() -> None:
    template = (
        '{"data": []}\nrouter={router_level}\nscore={information_score}\n'
        "Topic segment:\n{segment_text}"
    )
    rendered = render_extraction_prompt(template, "medium", [_segment()])
    assert '{"data": []}' in rendered
    assert "router=medium" in rendered
    assert "--- Topic sample-1:nsp_text_tiling:seg_000001 ---" in rendered
    assert "Alice: durable fact 1" in rendered
    assert "{segment_text}" not in rendered


def test_buffer_flushes_and_never_mixes_samples() -> None:
    batches = []
    template = "{router_level} {information_score}\n{segment_text}"
    buffer = ExtractionBuffer("small", template, 2, 100, 120, 5, lambda value: len(value.split()), lambda _t, _s, items, _p: batches.append(len(items)))
    for index in range(1, 4):
        buffer.add(_segment(index))
    buffer.finalize()
    assert batches == [2, 1]
    buffer.add(_segment())
    with pytest.raises(ValueError, match="cannot mix samples"):
        buffer.add(_segment(2, "other"))


def test_buffer_rejects_an_unprepared_segment_over_total_context() -> None:
    template = "{router_level} {information_score}\n{segment_text}"
    buffer = ExtractionBuffer("large", template, 8, 2, 3, 1, lambda value: len(value.split()), lambda *_: None)
    with pytest.raises(OversizeSegmentError, match="oversize_segment"):
        buffer.add(_segment())


def test_input_oversize_segment_within_total_budget_runs_as_singleton() -> None:
    template = "{router_level} {information_score}\n{segment_text}"
    batches = []
    buffer = ExtractionBuffer(
        "small",
        template,
        6,
        20,
        60,
        5,
        lambda value: len(value.split()),
        lambda _tier, _sample, items, _prompt: batches.append(
            [item.segment_id for item in items]
        ),
        allow_oversize_singleton=True,
    )
    first = _segment(1)
    oversized = replace(
        _segment(2),
        text=" ".join(f"detail-{index}" for index in range(30)),
    )
    last = _segment(3)
    buffer.add(first)
    buffer.add(oversized)
    buffer.add(last)
    buffer.finalize()
    assert batches == [
        [first.segment_id],
        [oversized.segment_id],
        [last.segment_id],
    ]

    config = {
        "max_facts_per_segment": 15,
        "reserve_output_tokens_per_segment": 5,
        "allow_oversize_singleton": True,
        "buffers": {
            "small": {
                "max_segments": 6,
                "max_input_tokens": 20,
                "max_total_context_tokens": 60,
            }
        },
    }
    prepared, plan = prepare_extraction_segments(
        segments=[oversized],
        prompts={"small": template},
        extraction_config=config,
        token_counters={"small": lambda value: len(value.split())},
        tiers=("small",),
    )
    assert prepared == [oversized]
    assert plan == {"small": {}}


def test_segment_over_total_context_is_tail_truncated_once() -> None:
    segment = replace(
        _segment(),
        text=(
            "[2026-01-01T00:00:00.000, Thu] 0.Alice: short durable fact\n"
            "[2026-01-02T00:00:00.000, Fri] 1.Bob: "
            + " ".join(f"detail-{index}" for index in range(250))
            + "\n[2026-01-03T00:00:00.000, Sat] 2.Carol: trailing durable fact"
        ),
        token_count=275,
        start_turn=1,
        end_turn=3,
        turn_ids=(1, 2, 3),
        start_timestamp="2026-01-01T00:00:00.000",
        end_timestamp="2026-01-03T00:00:00.000",
    )
    prompt = "{router_level} {information_score}\n{segment_text}"
    config = {
        "max_facts_per_segment": 15,
        "reserve_output_tokens_per_segment": 20,
        "allow_oversize_singleton": True,
        "truncate_over_total_context": True,
        "buffers": {
            "small": {
                "max_segments": 6,
                "max_input_tokens": 140,
                "max_total_context_tokens": 180,
            }
        },
    }
    counter = lambda value: len(value.split())
    prepared, plan = prepare_extraction_segments(
        segments=[segment],
        prompts={"small": prompt},
        extraction_config=config,
        token_counters={"small": counter},
        tiers=("small",),
    )
    assert len(prepared) == 1
    truncated = prepared[0]
    assert truncated.segment_id == segment.segment_id
    assert truncated.source_content_hash == segment.source_content_hash
    assert truncated.turn_ids == segment.turn_ids
    assert truncated.start_turn == segment.start_turn
    assert truncated.end_turn == segment.end_turn
    assert truncated.start_timestamp == segment.start_timestamp
    assert truncated.end_timestamp == segment.end_timestamp
    assert truncated.extraction_truncated is True
    assert truncated.extraction_retained_char_count < truncated.extraction_original_char_count
    assert truncated.text.endswith("[TRUNCATED_TO_FIT_CONTEXT: trailing content omitted]")
    assert plan["small"][segment.segment_id]["strategy"] == "tail_truncation"
    assert plan["small"][segment.segment_id]["dropped_char_count"] > 0
    visible = set(truncated.extraction_visible_source_ids)
    assert visible
    assert visible < {0, 1, 2}
    assert _allowed_source_ids(truncated) == visible
    assert plan["small"][segment.segment_id]["visible_source_ids"] == sorted(visible)
    assert plan["small"][segment.segment_id]["dropped_source_ids"] == sorted(
        {0, 1, 2} - visible
    )
    metadata = _extraction_truncation_metadata(truncated)
    assert metadata["extraction_visible_source_ids"] == sorted(visible)
    assert metadata["extraction_dropped_source_ids"] == sorted({0, 1, 2} - visible)
    invalid_source = next(iter({0, 1, 2} - visible))
    with pytest.raises(ValueError, match="source_id .* does not belong"):
        parse_fact_batch(
            json.dumps(
                {
                    "processed_segment_ids": [segment.segment_id],
                    "data": [
                        {
                            "segment_id": segment.segment_id,
                            "source_ids": [invalid_source],
                            "fact": "A fact from content the model did not receive.",
                        }
                    ],
                }
            ),
            [segment.segment_id],
            15,
            {segment.segment_id: _allowed_source_ids(truncated)},
        )
    assert counter(render_extraction_prompt(prompt, "small", prepared)) + 20 <= 180
    assert _source_metadata(truncated, 0)["source_speaker"] == "Alice"

    model = ModelSpec("api", "openai_compatible", "small", "small", 4096, 1, "n/a", 1024)
    planned = estimate_candidate_plan(
        segments=prepared,
        prompts={"small": prompt},
        extraction_config=config,
        token_counters={"small": counter},
        models={"small": model},
        prices={"small": PriceSpec(1.0, 2.0)},
    )
    assert planned["small"]["batch_count"] == 1


def test_cost_conservation_and_virtual_rebatching() -> None:
    price = PriceSpec(1.0, 2.0)
    allocations = allocate_batch(ProviderUsage(101, 23, "model"), ["a", "b"], [1, 2], [2, 1], [1, 0], price)
    assert sum(item.input_tokens for item in allocations) == 101
    assert sum(item.output_tokens for item in allocations) == 23
    segments = [_segment(1), _segment(2), _segment(3)]
    history = {(item.segment_id, tier): ReplaySegmentCost(item.segment_id, tier, 10, 2) for item in segments for tier in ("small", "medium", "large")}
    cfg = {tier: {"max_segments": 8, "max_input_tokens": 100, "max_total_context_tokens": 120} for tier in ("small", "medium", "large")}
    result = replay_virtual_cost(segments, ["small", "large", "small"], history, cfg, {tier: price for tier in cfg}, {tier: 5 for tier in cfg})
    assert result.batch_count_by_tier == {"small": 1, "large": 1}
    assert result.input_tokens == 40
    assert normalize_virtual_cost(1.0, 1.0, 3.0) == 0.0
    assert normalize_virtual_cost(3.0, 1.0, 3.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="All-Large"):
        normalize_virtual_cost(1.0, 1.0, 1.0)


def test_replay_history_selects_latest_complete_extraction_run(tmp_path) -> None:
    segments = [_segment(1), _segment(2)]
    path = tmp_path / "candidate_ledger.sqlite3"
    ledger = SqliteLedger(
        path,
        "segment_costs",
        ("extraction_run_id", "batch_id", "segment_id"),
    )
    for run_id in ("old", "new"):
        for segment in segments:
            for tier in ("small", "medium", "large"):
                ledger.append(
                    {
                        "extraction_run_id": run_id,
                        "batch_id": f"{run_id}:{tier}:{segment.segment_id}",
                        "segment_id": segment.segment_id,
                        "tier": tier,
                        "serialized_input_tokens": 10,
                        "attributed_output_tokens": 2,
                        "status": "ok",
                    }
                )
    run_id, history = load_replay_history(path, segments)
    assert run_id == "new"
    assert len(history) == 6
    selected, _ = load_replay_history(path, segments, "old")
    assert selected == "old"


def test_replay_history_rejects_duplicate_segment_costs(tmp_path) -> None:
    segment = _segment()
    rows = []
    for tier in ("small", "medium", "large"):
        for _ in range(2):
            rows.append(
                {
                    "extraction_run_id": "duplicated",
                    "segment_id": segment.segment_id,
                    "tier": tier,
                    "serialized_input_tokens": 10,
                    "attributed_output_tokens": 2,
                    "status": "ok",
                }
            )
    path = tmp_path / "segment_costs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate replay cost record"):
        load_replay_history(path, [segment])


def test_qdrant_filter_isolation_and_real_s_assembly() -> None:
    store = FactQdrantStore("unused", "test", 4, in_memory=True)
    first, second = _segment(1), _segment(2)
    for tier, value in (("small", 1.0), ("medium", 2.0), ("large", 3.0)):
        store.upsert_facts(tier, [_fact(first, tier), _fact(second, tier)], np.full((2, 4), value, dtype=np.float32))
    assembly = store.assemble(dataset_name="locomo", split="train", sample_id="sample-1", segments=[first, second], actions=["small", "large"], episode_id="ep", policy_version="test")
    assert assembly.status == "ready"
    points = store.assembly_points(dataset_name="locomo", split="train", sample_id="sample-1", assembly_id=assembly.assembly_id)
    assert {point.payload["source_collection_tier"] for point in points} == {"small", "large"}
    with pytest.raises(ValueError, match="assembly_id"):
        store.count_sample("assembled", dataset_name="locomo", split="train", sample_id="sample-1")
    assert len(store.search_assembly(np.ones(4), dataset_name="locomo", split="train", sample_id="sample-1", assembly_id=assembly.assembly_id, top_k=5)) == 2
    store.delete_assembly(dataset_name="locomo", split="train", sample_id="sample-1", assembly_id=assembly.assembly_id)
    assert store.assembly_points(dataset_name="locomo", split="train", sample_id="sample-1", assembly_id=assembly.assembly_id) == []
    store.close()


def test_reconciliation_detects_ghost_and_batch_replace_repairs_it(tmp_path) -> None:
    run_id = "reconcile-run"
    segment = _segment()
    namespace = "reconcile_namespace"
    store = FactQdrantStore("unused", namespace, 4, in_memory=True)
    run_dir = tmp_path / "outputs" / "rl_router" / "runs" / run_id
    state = ExtractionRunState(run_dir, run_id)
    state.register_run("scope", resume=False)
    ledger = SqliteLedger(
        tmp_path
        / "outputs"
        / "rl_router"
        / "locomo"
        / "train"
        / segment.segmentation_method
        / "samples"
        / segment.sample_id
        / "extraction"
        / "candidate_ledger.sqlite3",
        "segment_costs",
        ("extraction_run_id", "batch_id", "segment_id"),
    )
    facts = {}
    for sequence, tier in enumerate(("small", "medium", "large")):
        batch_id = f"batch-{tier}"
        state.plan_batch(
            batch_id=batch_id,
            sample_id=segment.sample_id,
            tier=tier,
            sequence_index=sequence,
            segment_ids=[segment.segment_id],
            prompt_hash=f"prompt-{tier}",
        )
        state.mark(batch_id, "committed")
        ledger.append(
            {
                "extraction_run_id": run_id,
                "batch_id": batch_id,
                "segment_id": segment.segment_id,
                "tier": tier,
                "fact_count": 1,
                "serialized_input_tokens": 10,
                "attributed_output_tokens": 2,
                "status": "ok",
            }
        )
        fact = replace(_fact(segment, tier, run_id), batch_id=batch_id)
        facts[tier] = fact
        store.upsert_facts(tier, [fact], np.ones((1, 4), dtype=np.float32))
    state.finish_run("complete")
    state.close()
    manifest = {
        "extraction_run_id": run_id,
        "status": "complete",
        "dataset_name": "locomo",
        "split": "train",
        "sample_id": segment.sample_id,
        "segmentation_method": segment.segmentation_method,
        "qdrant_collection_namespace": namespace,
        "required_tiers": ["small", "medium", "large"],
        "completed_tiers": ["small", "medium", "large"],
        "planned_extraction": {
            tier: {"batch_count": 1} for tier in ("small", "medium", "large")
        },
        "extraction_summary": {
            "batch_status_by_tier": {
                tier: {"committed": 1} for tier in ("small", "medium", "large")
            },
            "fact_counts": {tier: 1 for tier in ("small", "medium", "large")},
        },
        "qdrant_audit": {
            "counts_by_tier": {tier: 1 for tier in ("small", "medium", "large")}
        },
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    assert reconcile_extraction_run(tmp_path, run_id, store)["passed"] is True

    canonical = facts["small"]
    ghost = replace(
        canonical,
        fact_id=str(uuid.uuid4()),
        fact_text="stale response fact",
    )
    store.upsert_facts("small", [ghost], np.zeros((1, 4), dtype=np.float32))
    broken = reconcile_extraction_run(tmp_path, run_id, store)
    assert broken["passed"] is False
    assert any("segment-ledger/Qdrant" in error for error in broken["errors"])

    store.replace_candidate_batch(
        "small",
        dataset_name="locomo",
        split="train",
        sample_id=segment.sample_id,
        extraction_run_id=run_id,
        batch_id=canonical.batch_id,
        facts=[canonical],
        vectors=np.ones((1, 4), dtype=np.float32),
    )
    assert reconcile_extraction_run(tmp_path, run_id, store)["passed"] is True
    store.close()


def test_assembly_can_pin_one_candidate_extraction_run() -> None:
    store = FactQdrantStore("unused", "run_filter", 4, in_memory=True)
    segment = _segment()
    store.upsert_facts(
        "small",
        [_fact(segment, "small", "run-1"), _fact(segment, "small", "run-2")],
        np.ones((2, 4), dtype=np.float32),
    )
    assert len(
        store.candidate_points(
            "small",
            dataset_name="locomo",
            split="train",
            sample_id="sample-1",
            extraction_run_id="run-1",
        )
    ) == 1
    assembly = store.assemble(
        dataset_name="locomo",
        split="train",
        sample_id="sample-1",
        segments=[segment],
        actions=["small"],
        episode_id="ep",
        policy_version="test",
        extraction_run_id="run-1",
    )
    points = store.assembly_points(
        dataset_name="locomo",
        split="train",
        sample_id="sample-1",
        assembly_id=assembly.assembly_id,
    )
    assert [point.payload["extraction_run_id"] for point in points] == ["run-1"]
    store.close()


class _FakeEncoder:
    model_name = "fake-encoder"
    dimension = 4

    def encode(self, texts):
        return np.ones((len(texts), self.dimension), dtype=np.float32)


def test_candidate_generation_repairs_json_audits_and_resumes_without_new_calls(tmp_path) -> None:
    segment = replace(
        _segment(),
        text="[2026-01-01T00:00:00.000, Thu] 0.Alice: I moved to Paris.",
    )
    models = {
        tier: ModelSpec("api", "openai_compatible", tier, tier, 4096, 1, "n/a", 1024)
        for tier in ("small", "medium", "large")
    }
    prices = {tier: PriceSpec(1.0, 2.0) for tier in models}
    prompts = {
        tier: "router={router_level}\nscore={information_score}\n{segment_text}"
        for tier in models
    }
    calls = []

    def complete(tier, prompt, max_new_tokens):
        calls.append((tier, prompt, max_new_tokens))
        if "deterministic JSON repair tool" not in prompt:
            content = '{"processed_segment_ids": [}'
        else:
            content = json.dumps(
                {
                    "processed_segment_ids": [segment.segment_id],
                    "data": [
                        {
                            "segment_id": segment.segment_id,
                            "source_id": 0,
                            "fact": "Alice moved to Paris.",
                        }
                    ],
                }
            )
        return BatchCompletion(content, ProviderUsage(20, 5, tier, "provider", 0, 3))

    config = {
        "max_facts_per_segment": 10,
        "reserve_output_tokens_per_segment": 64,
        "schema_repair_max_attempts": 2,
        "require_provider_usage": True,
        "buffers": {
            tier: {
                "max_segments": 8,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
            for tier in models
        },
    }
    store = FactQdrantStore("unused", "repair_resume", 4, in_memory=True)
    progress_events = []
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models=models,
        prices=prices,
        token_counters={tier: lambda text: len(text.split()) for tier in models},
        completion=complete,
        prompts=prompts,
        prompt_versions={tier: "test-v1" for tier in models},
        extraction_config=config,
        output_root=tmp_path,
        progress_callback=progress_events.append,
    )
    first = generator.generate([segment], "repair-run")
    assert first.status == "complete"
    assert first.fact_counts == {"small": 1, "medium": 1, "large": 1}
    assert first.attempt_summary["repair_calls"] == 3
    assert len(calls) == 6
    assert len(progress_events) == 3
    assert {event["tier"] for event in progress_events} == set(models)
    assert all(event["status"] == "committed" for event in progress_events)
    assert all(event["fact_count"] == 1 for event in progress_events)
    assert all(event["logical_calls"] == 2 for event in progress_events)

    resumed = generator.generate([segment], "repair-run", resume=True)
    assert resumed.status == "complete"
    assert resumed.fact_counts == first.fact_counts
    assert len(calls) == 6
    assert len(progress_events) == 6

    with pytest.raises(ValueError, match="already exists"):
        generator.generate([segment], "repair-run")

    export_path = export_memories(
        store,
        "small",
        dataset_name="locomo",
        split="train",
        sample_id="sample-1",
        extraction_run_id="repair-run",
        output_path=tmp_path / "runs" / "repair-run" / "human_readable" / "L.json",
    )
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["metadata"]["extraction_run_id"] == "repair-run"
    assert len(exported["memories"]) == 1
    with pytest.raises(ValueError, match="requires extraction_run_id"):
        export_memories(
            store,
            "small",
            dataset_name="locomo",
            split="train",
            sample_id="sample-1",
            output_path=tmp_path / "unsafe.json",
        )
    store.close()


def test_candidate_tiers_can_run_independently_under_one_run_id(tmp_path) -> None:
    segment = replace(
        _segment(),
        text="[2026-01-01T00:00:00.000, Thu] 0.Alice: I moved to Paris.",
        end_turn=2,
        turn_ids=(1, 2),
        end_timestamp="2026-01-02T00:00:00.000",
        extraction_truncated=True,
        extraction_original_char_count=100,
        extraction_retained_char_count=70,
        extraction_visible_source_ids=(0,),
    )
    tiers = ("small", "medium", "large")
    models = {
        tier: ModelSpec("api", "openai_compatible", tier, tier, 4096, 1, "n/a", 1024)
        for tier in tiers
    }
    calls = []

    def complete(tier, prompt, max_new_tokens):
        calls.append(tier)
        return BatchCompletion(
            json.dumps(
                {
                    "processed_segment_ids": [segment.segment_id],
                    "data": [
                        {
                            "segment_id": segment.segment_id,
                            "source_ids": [0],
                            "fact": f"Alice moved to Paris according to {tier}.",
                        }
                    ],
                }
            ),
            ProviderUsage(20, 5, tier),
        )

    config = {
        "max_facts_per_segment": 10,
        "reserve_output_tokens_per_segment": 64,
        "schema_repair_max_attempts": 2,
        "require_provider_usage": True,
        "buffers": {
            tier: {
                "max_segments": 8,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
            for tier in tiers
        },
    }
    store = FactQdrantStore("unused", "independent_tiers", 4, in_memory=True)
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models=models,
        prices={tier: PriceSpec(1.0, 2.0) for tier in tiers},
        token_counters={tier: lambda text: len(text.split()) for tier in tiers},
        completion=complete,
        prompts={tier: "{router_level}\n{information_score}\n{segment_text}" for tier in tiers},
        prompt_versions={tier: "test-v1" for tier in tiers},
        extraction_config=config,
        output_root=tmp_path,
    )

    first = generator.generate([segment], "tier-run", tiers=("small",))
    assert first.selected_tiers == ["small"]
    assert first.fact_counts == {"small": 1, "medium": 0, "large": 0}
    second = generator.generate(
        [segment], "tier-run", resume=True, tiers=("medium",)
    )
    assert second.selected_tiers == ["medium"]
    third = generator.generate(
        [segment], "tier-run", resume=True, tiers=("large",)
    )
    assert third.fact_counts == {"small": 1, "medium": 1, "large": 1}
    assert calls == ["small", "medium", "large"]
    for tier in tiers:
        points = store.candidate_points(
            tier,
            dataset_name="locomo",
            split="train",
            sample_id="sample-1",
            extraction_run_id="tier-run",
        )
        assert len(points) == 1
        assert points[0].payload["source_ids"] == [0]
        assert len(points[0].payload["source_provenance"]) == 1
        assert points[0].payload["extraction_visible_source_ids"] == [0]
        assert points[0].payload["extraction_dropped_source_ids"] == [1]
        assert points[0].payload["segment_end_timestamp"] == segment.end_timestamp
    candidate_ledger_path = (
        tmp_path / "locomo" / "train" / segment.segmentation_method / "samples"
        / segment.sample_id / "extraction" / "candidate_ledger.sqlite3"
    )
    cost_rows = read_sqlite_ledger(candidate_ledger_path, "segment_costs")
    assert len(cost_rows) == 3
    assert all(row["allocated_input_tokens"] == 20 for row in cost_rows)
    assert all(row["allocated_output_tokens"] == 5 for row in cost_rows)
    assert all(row["extraction_visible_source_ids"] == [0] for row in cost_rows)
    assert all(row["extraction_dropped_source_ids"] == [1] for row in cost_rows)
    store.close()


def test_segment_audit_keeps_zero_fact_segments_and_full_provenance(tmp_path) -> None:
    segments = [
        replace(
            _segment(index),
            text=(
                f"[2026-01-0{index}T00:00:00.000, Thu] "
                f"{index - 1}.Alice: source {index}."
            ),
            turn_ids=(index,),
            token_count=10 + index,
        )
        for index in (1, 2)
    ]
    prompt = "{router_level}\n{information_score}\n{segment_text}"
    model = ModelSpec(
        deploy="api",
        backend="openai_compatible",
        model_name="configured-small",
        tokenizer_name="small",
        max_context_tokens=4096,
        tensor_parallel_size=1,
        dtype="n/a",
        max_output_tokens=1024,
        api_base_url="https://provider.example/v1",
        request_model_name="resolved-small",
    )
    config = {
        "max_facts_per_segment": 10,
        "reserve_output_tokens_per_segment": 64,
        "schema_repair_max_attempts": 0,
        "require_provider_usage": True,
        "buffers": {
            "small": {
                "max_segments": 2,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
        },
    }

    def complete(tier, _rendered, _max_new_tokens):
        return BatchCompletion(
            json.dumps(
                {
                    "processed_segment_ids": [item.segment_id for item in segments],
                    "data": [
                        {
                            "segment_id": segments[0].segment_id,
                            "source_ids": [0],
                            "fact": "Only the first segment produced a fact.",
                        }
                    ],
                }
            ),
            ProviderUsage(
                30,
                8,
                "resolved-small",
                latency_ms=17,
                provider_request_id="request-1",
            ),
        )

    store = FactQdrantStore("unused", "qwen_audit", 4, in_memory=True)
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models={"small": model},
        prices={"small": PriceSpec(1.0, 2.0, "USD", "2026-08-08")},
        token_counters={"small": lambda text: len(text.split())},
        completion=complete,
        prompts={"small": prompt},
        prompt_versions={"small": "test-v1"},
        extraction_config=config,
        output_root=tmp_path,
        audit_context={
            "model_family": "qwen",
            "campaign_id": "qwen-campaign",
            "campaign_scope_hash": "campaign-scope",
            "qdrant_namespace": "qwen_audit",
            "qdrant_distance": "Cosine",
            "embedding_model_hash": "embedding-hash",
            "embedding_revision": "embedding-revision",
            "embedding_normalized": True,
        },
    )
    summary = generator.generate(segments, "audit-run", tiers=("small",))
    assert summary.status == "complete"
    ledger_path = (
        tmp_path
        / "locomo"
        / "train"
        / segments[0].segmentation_method
        / "samples"
        / segments[0].sample_id
        / "extraction"
        / "candidate_ledger.sqlite3"
    )
    rows = read_sqlite_ledger(ledger_path, "segment_costs")
    assert len(rows) == 2
    by_segment = {row["segment_id"]: row for row in rows}
    assert by_segment[segments[0].segment_id]["fact_count"] == 1
    assert by_segment[segments[0].segment_id]["status"] == "ok"
    assert by_segment[segments[1].segment_id]["fact_count"] == 0
    assert by_segment[segments[1].segment_id]["status"] == "no_fact"
    zero = by_segment[segments[1].segment_id]
    assert zero["audit_schema_version"] == "segment_extraction_audit_v1"
    assert zero["model_family"] == "qwen"
    assert zero["campaign_id"] == "qwen-campaign"
    assert zero["segment_turn_count"] == 1
    assert zero["segment_token_count"] == 12
    assert zero["extractor_request_model"] == "resolved-small"
    assert zero["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert zero["batch_logical_call_count"] == 1
    assert zero["batch_latency_ms"] == 17
    assert zero["batch_provider_request_ids"] == ["request-1"]

    points = store.candidate_points(
        "small",
        dataset_name="locomo",
        split="train",
        sample_id=segments[0].sample_id,
        extraction_run_id="audit-run",
        with_vectors=False,
    )
    assert len(points) == 1
    assert points[0].payload["schema_version"] == "qdrant_fact_v3"
    assert points[0].payload["model_family"] == "qwen"
    assert points[0].payload["campaign_id"] == "qwen-campaign"
    store.close()


def test_routed_generation_extracts_each_segment_once_in_only_its_selected_tier(tmp_path) -> None:
    tiers = ("small", "medium", "large")
    segments = [
        replace(
            _segment(index),
            text=f"[2026-01-01T00:00:00.000, Thu] {index - 1}.Alice: durable fact {index}.",
            turn_ids=(index,),
        )
        for index in range(1, 4)
    ]
    actions = ["small", "large", "small"]
    models = {
        tier: ModelSpec("api", "openai_compatible", tier, tier, 4096, 1, "n/a", 1024)
        for tier in tiers
    }
    prompts = {tier: "{router_level}\n{information_score}\n{segment_text}" for tier in tiers}
    config = {
        "max_facts_per_segment": 10,
        "reserve_output_tokens_per_segment": 64,
        "schema_repair_max_attempts": 0,
        "require_provider_usage": True,
        "buffers": {
            tier: {
                "max_segments": 1,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
            for tier in tiers
        },
    }
    calls = []

    def complete(tier, prompt, _max_new_tokens):
        segment = next(item for item in segments if item.segment_id in prompt)
        calls.append((segment.segment_id, tier))
        return BatchCompletion(
            json.dumps(
                {
                    "processed_segment_ids": [segment.segment_id],
                    "data": [
                        {
                            "segment_id": segment.segment_id,
                            "source_ids": [segment.turn_ids[0] - 1],
                            "fact": f"A standalone fact extracted by {tier}.",
                        }
                    ],
                }
            ),
            ProviderUsage(10, 4, tier),
        )

    counters = {tier: lambda text: len(text.split()) for tier in tiers}
    prepared, truncation = prepare_routed_extraction_segments(
        segments=segments,
        actions=actions,
        prompts=prompts,
        extraction_config=config,
        token_counters=counters,
    )
    assert prepared == segments
    assert truncation == {"small": {}, "large": {}}
    plan = estimate_routed_plan(
        segments=prepared,
        actions=actions,
        prompts=prompts,
        extraction_config=config,
        token_counters=counters,
        models=models,
        prices={tier: PriceSpec(1.0, 2.0) for tier in tiers},
    )
    assert {tier: plan[tier]["segment_count"] for tier in tiers} == {
        "small": 2,
        "medium": 0,
        "large": 1,
    }

    store = FactQdrantStore("unused", "routed", 4, in_memory=True)
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models=models,
        prices={tier: PriceSpec(1.0, 2.0) for tier in tiers},
        token_counters=counters,
        completion=complete,
        prompts=prompts,
        prompt_versions={tier: "test-v1" for tier in tiers},
        extraction_config=config,
        output_root=tmp_path,
        ledger_filename="deployment_ledger.sqlite3",
    )
    summary = generator.generate_routed(
        prepared,
        actions,
        "deployment-run",
        route_scope={"fold": 0, "checkpoint_sha256": "abc"},
    )
    assert summary.status == "complete"
    assert calls == [
        (segments[0].segment_id, "small"),
        (segments[2].segment_id, "small"),
        (segments[1].segment_id, "large"),
    ]
    assert summary.fact_counts == {"small": 2, "medium": 0, "large": 1}
    assert summary.attempt_summary["logical_api_calls"] == 3
    assert summary.attempt_summary["provider_input_tokens"] == 30
    for segment, selected_tier in zip(segments, actions):
        for tier in tiers:
            points = store.candidate_points(
                tier,
                dataset_name=segment.dataset_name,
                split=segment.split,
                sample_id=segment.sample_id,
                segment_id=segment.segment_id,
                extraction_run_id="deployment-run",
            )
            assert len(points) == (1 if tier == selected_tier else 0)
    resumed = generator.generate_routed(
        prepared,
        actions,
        "deployment-run",
        resume=True,
        route_scope={"fold": 0, "checkpoint_sha256": "abc"},
    )
    assert resumed.fact_counts == summary.fact_counts
    assert len(calls) == 3
    store.close()


def test_deployment_cost_summary_reports_both_denominators() -> None:
    summary = CandidateGenerationSummary(
        sample_id="sample-1",
        extraction_run_id="run",
        fact_counts={"small": 1, "medium": 0, "large": 0},
        batch_counts={"small": 1, "medium": 0, "large": 0},
        batch_status_counts={"committed": 1},
        batch_status_by_tier={"small": {"committed": 1}, "medium": {}, "large": {}},
        selected_tiers=["small"],
        status="complete",
        known_cost=0.6,
        unknown_cost_attempts=0,
        attempt_summary={
            "logical_api_calls": 2,
            "successful_attempts": 2,
            "failed_attempts": 0,
            "repair_calls": 1,
            "provider_input_tokens": 100,
            "provider_output_tokens": 20,
        },
        quality_metrics={},
    )
    result = summarize_deployment_costs([summary], question_count=3)
    assert result["per_sample"]["cost"] == pytest.approx(0.6)
    assert result["per_question"]["amortized_extraction_cost"] == pytest.approx(0.2)
    assert result["per_extraction_call"]["input_tokens"] == pytest.approx(50)
    assert validate_route_decisions([_segment()], ["small"])
    with pytest.raises(ValueError, match="exactly one action"):
        validate_route_decisions([_segment()], [])


def test_question_metrics_match_lightmem_category_and_population_std() -> None:
    questions = [
        {
            "question_id": "q1",
            "category": "category_1",
            "question_type": "single-hop",
        },
        {
            "question_id": "q2",
            "category": "category_1",
            "question_type": "single-hop",
        },
        {
            "question_id": "q3",
            "category": "category_2",
            "question_type": "temporal",
        },
        {
            "question_id": "q4",
            "category": "category_2",
            "question_type": "temporal",
        },
    ]
    evaluations = [
        SimpleNamespace(question_id="q1", correct=True),
        SimpleNamespace(question_id="q2", correct=False),
        SimpleNamespace(question_id="q3", correct=True),
        SimpleNamespace(question_id="q4", correct=True),
    ]
    outcomes = build_question_outcomes(questions, evaluations)
    result = summarize_question_outcomes(outcomes)

    assert result["overall"]["judge_correct"] == {
        "mean": 0.75,
        "std": pytest.approx(np.sqrt(0.75 * 0.25)),
        "count": 4,
    }
    assert result["by_category"]["category_1"]["accuracy"] == 0.5
    assert result["by_category"]["category_2"]["accuracy"] == 1.0
    assert result["category_distribution"]["category_1"]["fraction"] == 0.5
    assert result["by_question_type"]["temporal"]["count"] == 2


def test_qa_usage_accepts_ledger_dicts_for_resume_backfill() -> None:
    result = summarize_qa_usage(
        [
            {
                "reader_input_tokens": 100,
                "reader_output_tokens": 10,
                "reader_input_cost": 0.1,
                "reader_output_cost": 0.02,
                "reader_retry_count": 1,
                "judge_input_tokens": 20,
                "judge_output_tokens": 2,
                "judge_input_cost": 0.03,
                "judge_output_cost": 0.01,
                "judge_retry_count": 0,
            }
        ]
    )
    assert result["reader"]["total_tokens"] == 110
    assert result["reader"]["transport_attempts"] == 2
    assert result["judge"]["total_tokens"] == 22
    assert result["total_tokens"] == 132


def test_candidate_generation_continues_after_transport_failure_and_resume_finishes(tmp_path) -> None:
    segment = replace(
        _segment(),
        text="[2026-01-01T00:00:00.000, Thu] 0.Alice: I moved to Paris.",
    )
    tiers = ("small", "medium", "large")
    models = {
        tier: ModelSpec("api", "openai_compatible", tier, tier, 4096, 1, "n/a", 1024)
        for tier in tiers
    }
    failed_once = {"small": False}

    def complete(tier, prompt, max_new_tokens):
        if tier == "small" and not failed_once["small"]:
            failed_once["small"] = True
            raise ModelAPIError(
                "timeout",
                attempts=[
                    {
                        "transport_attempt": 1,
                        "status": "failed",
                        "http_status": None,
                        "latency_ms": 5,
                        "cost_status": "unknown",
                        "error": "timeout",
                    },
                    {
                        "transport_attempt": 2,
                        "status": "failed",
                        "http_status": None,
                        "latency_ms": 5,
                        "cost_status": "unknown",
                        "error": "timeout",
                    },
                ],
            )
        return BatchCompletion(
            json.dumps(
                {
                    "processed_segment_ids": [segment.segment_id],
                    "data": [],
                }
            ),
            ProviderUsage(20, 5, tier, "provider", 0, 3),
        )

    config = {
        "max_facts_per_segment": 10,
        "reserve_output_tokens_per_segment": 64,
        "schema_repair_max_attempts": 2,
        "require_provider_usage": True,
        "buffers": {
            tier: {
                "max_segments": 8,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
            for tier in tiers
        },
    }
    store = FactQdrantStore("unused", "failure_resume", 4, in_memory=True)
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models=models,
        prices={tier: PriceSpec(1.0, 2.0) for tier in tiers},
        token_counters={tier: lambda text: len(text.split()) for tier in tiers},
        completion=complete,
        prompts={tier: "{router_level}\n{information_score}\n{segment_text}" for tier in tiers},
        prompt_versions={tier: "test-v1" for tier in tiers},
        extraction_config=config,
        output_root=tmp_path,
    )
    interrupted = generator.generate([segment], "failure-run")
    assert interrupted.status == "incomplete"
    assert interrupted.batch_status_counts == {"committed": 2, "failed_retryable": 1}
    assert interrupted.unknown_cost_attempts == 2

    resumed = generator.generate([segment], "failure-run", resume=True)
    assert resumed.status == "complete"
    assert resumed.batch_status_counts == {"committed": 3}
    assert resumed.unknown_cost_attempts == 2
    store.close()


def test_non_retryable_provider_error_opens_tier_circuit(tmp_path) -> None:
    segment = _segment()
    model = ModelSpec("api", "openai_compatible", "small", "small", 4096, 1, "n/a", 1024)
    calls = []

    def complete(tier, prompt, max_new_tokens):
        calls.append(tier)
        raise ModelAPIError("HTTP 401", retryable=False)

    config = {
        "max_facts_per_segment": 15,
        "reserve_output_tokens_per_segment": 64,
        "require_provider_usage": True,
        "buffers": {
            "small": {
                "max_segments": 1,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
        },
    }
    store = FactQdrantStore("unused", "fatal_provider", 4, in_memory=True)
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models={"small": model},
        prices={"small": PriceSpec(1.0, 2.0)},
        token_counters={"small": lambda text: len(text.split())},
        completion=complete,
        prompts={"small": "{router_level}\n{information_score}\n{segment_text}"},
        prompt_versions={"small": "test-v1"},
        extraction_config=config,
        output_root=tmp_path,
    )
    with pytest.raises(ProviderCircuitOpenError, match="tier small"):
        generator.generate([segment, _segment(2)], "fatal-provider-run")
    assert calls == ["small"]
    store.close()


def test_finish_reason_length_is_terminal_and_quality_metrics_track_saturation(tmp_path) -> None:
    segment = _segment()
    model = ModelSpec("api", "openai_compatible", "small", "small", 4096, 1, "n/a", 1024)
    config = {
        "max_facts_per_segment": 1,
        "reserve_output_tokens_per_segment": 64,
        "require_provider_usage": True,
        "quality_gates": {
            "max_empty_fact_segment_rate": 1.0,
            "max_saturated_segment_rate": 0.0,
            "max_repair_batch_rate": 1.0,
            "max_failed_batch_rate": 1.0,
        },
        "buffers": {
            "small": {
                "max_segments": 1,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
        },
    }
    calls = {"finish": "length"}

    def complete(tier, prompt, max_new_tokens):
        return BatchCompletion(
            json.dumps(
                {
                    "processed_segment_ids": [segment.segment_id],
                    "data": [{"segment_id": segment.segment_id, "source_ids": [0], "fact": "Alice has a durable fact."}],
                }
            ),
            ProviderUsage(20, 5, tier, "provider", finish_reason=calls["finish"]),
        )

    store = FactQdrantStore("unused", "finish_reason", 4, in_memory=True)
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models={"small": model},
        prices={"small": PriceSpec(1.0, 2.0)},
        token_counters={"small": lambda text: len(text.split())},
        completion=complete,
        prompts={"small": "{router_level}\n{information_score}\n{segment_text}"},
        prompt_versions={"small": "test-v1"},
        extraction_config=config,
        output_root=tmp_path,
    )
    terminal = generator.generate([segment], "length-run")
    assert terminal.batch_status_counts == {"failed_terminal": 1}
    calls["finish"] = "stop"
    completed = generator.generate(
        [segment], "length-run", resume=True, retry_terminal=True
    )
    assert completed.status == "complete"
    assert completed.quality_metrics["saturated_segments"] == 1
    assert completed.quality_metrics["passed"] is False
    store.close()


def test_explicit_retry_terminal_starts_fresh_model_calls(tmp_path) -> None:
    segment = replace(
        _segment(),
        text="[2026-01-01T00:00:00.000, Thu] 0.Alice: I moved to Paris.",
    )
    tiers = ("small", "medium", "large")
    models = {
        tier: ModelSpec("api", "openai_compatible", tier, tier, 4096, 1, "n/a", 1024)
        for tier in tiers
    }
    fixed = {"value": False}
    call_count = {tier: 0 for tier in tiers}

    def complete(tier, prompt, max_new_tokens):
        call_count[tier] += 1
        item = {
            "segment_id": segment.segment_id,
            "source_id": 0 if fixed["value"] else 999,
            "fact": "Alice moved to Paris.",
        }
        return BatchCompletion(
            json.dumps({"processed_segment_ids": [segment.segment_id], "data": [item]}),
            ProviderUsage(20, 5, tier),
        )

    config = {
        "max_facts_per_segment": 10,
        "reserve_output_tokens_per_segment": 64,
        "schema_repair_max_attempts": 2,
        "require_provider_usage": True,
        "buffers": {
            tier: {
                "max_segments": 8,
                "max_input_tokens": 1000,
                "max_total_context_tokens": 2000,
            }
            for tier in tiers
        },
    }
    store = FactQdrantStore("unused", "terminal_retry", 4, in_memory=True)
    generator = CandidateGenerator(
        store=store,
        encoder=_FakeEncoder(),
        models=models,
        prices={tier: PriceSpec(1.0, 2.0) for tier in tiers},
        token_counters={tier: lambda text: len(text.split()) for tier in tiers},
        completion=complete,
        prompts={tier: "{router_level}\n{information_score}\n{segment_text}" for tier in tiers},
        prompt_versions={tier: "test-v1" for tier in tiers},
        extraction_config=config,
        output_root=tmp_path,
    )
    terminal = generator.generate([segment], "terminal-run")
    assert terminal.batch_status_counts == {"failed_terminal": 3}
    fixed["value"] = True
    completed = generator.generate(
        [segment], "terminal-run", resume=True, retry_terminal=True
    )
    assert completed.status == "complete"
    assert call_count == {"small": 2, "medium": 2, "large": 2}
    store.close()


def test_router_checkpoint_restores_policy_and_scaler(tmp_path) -> None:
    model, scaler = EmbeddingMLPRouter(10, [8], 0.0), FeatureScaler([0.0] * 6, [1.0] * 6)
    features = np.ones((3, 10), dtype=np.float32)
    before = [item.tier for item in model.route(features, deterministic=True)]
    model.save_checkpoint(tmp_path / "router.pt", scaler, {"seed": 42})
    restored, loaded_scaler, metadata = EmbeddingMLPRouter.load_checkpoint(tmp_path / "router.pt")
    assert [item.tier for item in restored.route(features, deterministic=True)] == before
    assert loaded_scaler == scaler and metadata["seed"] == 42


def test_judge_protocol() -> None:
    assert parse_judge_label("CORRECT\nreason") is True
    assert parse_judge_label("INCORRECT") is False
    with pytest.raises(ValueError):
        parse_judge_label("yes")


class _FakeCompletionClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(self.responses.pop(0), 11, 3, 7)


def test_lightmem_reader_renders_the_external_prompt_without_rewriting() -> None:
    bundle = load_rl_bundle("configs")
    client = _FakeCompletionClient(["Paris"])
    reader = LightMemQAReader(bundle, client)
    result = reader.answer(
        dataset_name="locomo",
        question={"question": "Where did Alice go?"},
        retrieved=[
            {
                "fact_text": "Alice visited Paris.",
                "source_speaker": "Alice",
                "source_timestamp": "2023-05-08T13:56:00.000",
                "source_weekday": "Mon",
            },
            {"fact_text": "Bob stayed home.", "source_speaker": "Bob"},
        ],
        sample_metadata={"speaker_a": "Alice", "speaker_b": "Bob"},
    )
    expected = bundle.prompt_path("locomo_answer").read_text(encoding="utf-8").format(
        speaker_1_name="Alice",
        speaker_1_memories="[2023-05-08T13:56:00.000, Mon] Alice visited Paris.",
        speaker_2_name="Bob",
        speaker_2_memories="Bob stayed home.",
        question="Where did Alice go?",
    )
    assert result.answer == "Paris"
    assert client.calls[0]["prompt"] == expected
    assert client.calls[0]["json_mode"] is False


@pytest.mark.parametrize(
    ("question_type", "profile", "unanswerable", "expected"),
    [
        ("single-session-user", "", False, "longmemeval_single_session_judge"),
        ("single-session-assistant", "", False, "longmemeval_single_session_judge"),
        ("multi-session", "", False, "longmemeval_single_session_judge"),
        ("temporal-reasoning", "", False, "longmemeval_temporal_reasoning_judge"),
        ("knowledge-update", "", False, "longmemeval_knowledge_update_judge"),
        ("single-session-preference", "", False, "longmemeval_preference_judge"),
        ("anything", "longmemeval_abstention", True, "longmemeval_abstention_judge"),
        ("anything", "", False, "longmemeval_exact_match_judge"),
    ],
)
def test_lightmem_longmemeval_judge_routing(question_type, profile, unanswerable, expected) -> None:
    assert select_judge_prompt(
        "longmemeval",
        question_type,
        judge_profile=profile,
        is_unanswerable=unanswerable,
    ) == expected


def test_lightmem_judges_use_exact_external_templates_and_protocols() -> None:
    bundle = load_rl_bundle("configs")
    locomo_client = _FakeCompletionClient(['{"label":"CORRECT"}'])
    locomo = LightMemLLMJudge(bundle, locomo_client)
    result = locomo.judge(
        dataset_name="locomo",
        question={"question": "Where?", "answer": "Paris", "question_id": "q1"},
        predicted_answer="Paris",
    )
    expected = (
        bundle.prompt_path("locomo_judge")
        .read_text(encoding="utf-8")
        .replace("$question", "Where?")
        .replace("$golden_answers", "Paris")
        .replace("$prediction", "Paris")
    )
    assert result.correct is True
    assert locomo_client.calls[0]["prompt"] == expected
    assert locomo_client.calls[0]["json_mode"] is True

    longmem_client = _FakeCompletionClient(["yes"])
    longmem = LightMemLLMJudge(bundle, longmem_client)
    result = longmem.judge(
        dataset_name="longmemeval",
        question={
            "question": "How many days?",
            "answer": "18 days",
            "question_type": "temporal-reasoning",
            "question_id": "q2",
        },
        predicted_answer="19 days",
    )
    assert result.correct is True
    assert "do not penalize off-by-one errors" in longmem_client.calls[0]["prompt"]
    assert longmem_client.calls[0]["json_mode"] is False


def test_lightmem_judge_response_parsers_reject_ambiguous_labels() -> None:
    assert parse_locomo_judge_label('{"label":"CORRECT"}') is True
    assert parse_locomo_judge_label('{"label":"WRONG"}') is False
    assert parse_longmemeval_judge_label("Yes") is True
    assert parse_longmemeval_judge_label("No") is False
    with pytest.raises(ValueError):
        parse_locomo_judge_label("CORRECT")
    with pytest.raises(ValueError):
        parse_longmemeval_judge_label("not sure")


def test_chat_completion_parser_supports_json_and_sse() -> None:
    ordinary = parse_chat_completion_response('{"choices":[{"message":{"content":"FACT"}}],"usage":{"prompt_tokens":1}}')
    assert ordinary["choices"][0]["message"]["content"] == "FACT"
    sse = parse_chat_completion_response(
        'data: {"choices":[{"delta":{"content":"FA"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"CT"}}],"usage":{"completion_tokens":1}}\n\n'
        'data: [DONE]'
    )
    assert sse["choices"][0]["message"]["content"] == "FACT"

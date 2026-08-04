"""Formal model-role configuration tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from infobudget.config import load_project_bundle
from infobudget.rl_router.config import load_rl_bundle, scan_config_secrets
from infobudget.schemas import ModelSpec


def test_five_api_roles_have_prices_and_environment_keys() -> None:
    bundle = load_project_bundle("configs")
    roles = {"small", "medium", "large", "qa_reader", "judge_llm"}
    assert roles == bundle.models.keys()
    for role in roles:
        spec = bundle.models[role]
        assert spec.deploy == "api"
        assert spec.api_key_env
        assert spec.model_name in bundle.prices
    assert not scan_config_secrets("configs")
    rl_bundle = load_rl_bundle("configs")
    assert rl_bundle.rl["extraction"]["max_facts_per_segment"] == 15
    assert rl_bundle.rl["extraction"]["reserve_output_tokens_per_segment"] == 1024
    assert rl_bundle.rl["extraction"]["allow_oversize_singleton"] is True
    assert rl_bundle.rl["extraction"]["truncate_over_total_context"] is True
    assert rl_bundle.rl["extraction"]["quality_gates"]["max_failed_batch_rate"] == 0.0
    assert len({tuple(values.items()) for values in rl_bundle.rl["extraction"]["buffers"].values()}) == 1
    assert {
        values["max_segments"]
        for values in rl_bundle.rl["extraction"]["buffers"].values()
    } == {6}
    storage = rl_bundle.rl["storage"]
    assert storage["mode"] == "server"
    assert storage["url"] == "http://127.0.0.1:6333"
    assert storage["grpc_port"] == 6334
    assert "{embedding_hash}" in storage["collection_namespace"]
    compose = yaml.safe_load(
        Path("deploy/qdrant/docker-compose.yml").read_text(encoding="utf-8")
    )
    assert compose["services"]["qdrant"]["ports"] == [
        "127.0.0.1:${QDRANT_HTTP_PORT:-6333}:6333",
        "127.0.0.1:${QDRANT_GRPC_PORT:-6334}:6334",
    ]
    prompt = rl_bundle.prompt_path("fact_extraction").read_text(encoding="utf-8")
    assert "Maximum output: 15 facts per topic segment." in prompt
    assert '"processed_segment_ids"' in prompt
    assert '"segment_id"' in prompt
    assert '"source_ids"' in prompt
    assert '"fact"' in prompt
    assert set(rl_bundle.rl["prompts"]) == {
        "fact_extraction",
        "locomo_answer",
        "longmemeval_answer",
        "locomo_judge",
        "longmemeval_single_session_judge",
        "longmemeval_temporal_reasoning_judge",
        "longmemeval_knowledge_update_judge",
        "longmemeval_preference_judge",
        "longmemeval_abstention_judge",
        "longmemeval_exact_match_judge",
    }

    limits = {
        "small": (32768, 8192, 24576),
        "medium": (262144, 16384, 245760),
        "large": (131072, 32000, 99072),
    }
    for tier, (context, output, input_at_max_output) in limits.items():
        spec = rl_bundle.project.models[tier]
        assert (spec.max_context_tokens, spec.max_output_tokens, spec.max_input_tokens) == (
            context,
            output,
            input_at_max_output,
        )


def test_model_spec_resolves_only_environment_key(monkeypatch) -> None:
    spec = ModelSpec(
        deploy="api",
        backend="openai_compatible",
        model_name="example-model",
        tokenizer_name="example-model",
        max_context_tokens=4096,
        tensor_parallel_size=1,
        dtype="n/a",
        api_key_env="EXAMPLE_MODEL_API_KEY",
    )
    monkeypatch.setenv("EXAMPLE_MODEL_API_KEY", "test-key")
    assert spec.resolved_api_key() == "test-key"

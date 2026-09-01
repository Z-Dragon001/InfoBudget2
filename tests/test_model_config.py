"""Formal model-role configuration tests."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from infobudget.config import load_env_file, load_project_bundle
from infobudget.rl_router.config import load_rl_bundle, scan_config_secrets
from infobudget.schemas import ModelSpec


def test_api_roles_have_credentials_and_known_price_scopes() -> None:
    bundle = load_project_bundle("configs")
    priced_roles = {"small", "medium", "large", "qa_reader", "judge_llm"}
    roles = {*priced_roles, "gold_fact_extractor"}
    assert roles == bundle.models.keys()
    for role in roles:
        spec = bundle.models[role]
        assert spec.deploy == "api"
        assert spec.api_key_env
        if role in priced_roles:
            assert spec.model_name in bundle.prices
    gold = bundle.models["gold_fact_extractor"]
    assert gold.backend == "cloudflare_responses"
    assert "{account_id}" in gold.api_base_url
    assert gold.model_name not in bundle.prices
    assert not scan_config_secrets("configs")
    rl_bundle = load_rl_bundle("configs")
    assert rl_bundle.rl["extraction"]["max_facts_per_segment"] == 15
    assert rl_bundle.rl["extraction"]["reserve_output_tokens_per_segment"] == 1024
    assert rl_bundle.rl["extraction"]["allow_oversize_singleton"] is True
    assert rl_bundle.rl["extraction"]["truncate_over_total_context"] is True
    assert rl_bundle.rl["extraction"]["quality_gates"]["max_failed_batch_rate"] == 0.0
    buffers = rl_bundle.rl["extraction"]["buffers"]
    assert {
        tier: values["max_segments"] for tier, values in buffers.items()
    } == {"small": 3, "medium": 6, "large": 6}
    assert {
        (values["max_input_tokens"], values["max_total_context_tokens"])
        for values in buffers.values()
    } == {(16384, 24576)}
    storage = rl_bundle.rl["storage"]
    assert rl_bundle.embeddings["router"]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert rl_bundle.embeddings["router"]["dimension"] == 384
    assert rl_bundle.embeddings["memory"]["dimension"] == 384
    assert rl_bundle.embeddings["router"]["long_text_strategy"] == "mean_pool_chunks"
    assert rl_bundle.embeddings["memory"]["long_text_strategy"] == "truncate"
    assert storage["vector_size"] == 384
    assert rl_bundle.rl["model_family"] == "qwen"
    assert {
        tier: rl_bundle.project.models[tier].stable_model_id
        for tier in ("small", "medium", "large")
    } == {
        "small": "qwen2.5-7b-instruct",
        "medium": "qwen2.5-32b-instruct",
        "large": "qwen2.5-72b-instruct",
    }
    assert storage["mode"] == "server"
    assert storage["url"] == "http://127.0.0.1:6333"
    assert storage["grpc_port"] == 6334
    assert rl_bundle.project.config.project.name == "InfoBudget"
    assert "{project_name}" in storage["collection_namespace"]
    assert "{embedding_hash}" in storage["collection_namespace"]
    assert "{model_family}" in storage["collection_namespace"]
    compose = yaml.safe_load(
        Path("deploy/qdrant/docker-compose.yml").read_text(encoding="utf-8")
    )
    assert compose["services"]["qdrant"]["ports"] == [
        "127.0.0.1:${QDRANT_HTTP_PORT:-6333}:6333",
        "127.0.0.1:${QDRANT_GRPC_PORT:-6334}:6334",
    ]
    extraction_prompts = {
        dataset: rl_bundle.fact_extraction_prompt_path(dataset).read_text(
            encoding="utf-8"
        )
        for dataset in ("locomo", "longmemeval")
    }
    for dataset, prompt in extraction_prompts.items():
        assert "Maximum output: 15 facts per topic segment." in prompt
        assert '"processed_segment_ids"' in prompt
        assert '"segment_id"' in prompt
        assert '"source_ids"' in prompt
        assert '"fact"' in prompt
        assert "external knowledge" in prompt
    assert rl_bundle.fact_extraction_prompt_version("locomo").endswith("_v10")
    assert rl_bundle.fact_extraction_prompt_version("longmemeval").endswith("_v7")
    assert "Personal Information and Fact Extractor" in extraction_prompts["locomo"]
    assert "temporary states" in extraction_prompts["locomo"]
    assert "{segment_text_with_source_constraints}" in extraction_prompts["locomo"]
    assert "Allowed source_ids for this topic only" in extraction_prompts["locomo"]
    assert "Knowledge updates" in extraction_prompts["longmemeval"]
    assert "Abstention support" in extraction_prompts["longmemeval"]
    assert set(rl_bundle.rl["prompts"]) == {
        "fact_extraction_locomo",
        "fact_extraction_longmemeval",
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
    assert spec.stable_model_id == "example-model"


def test_project_env_file_loads_without_overriding_process_environment(
    tmp_path, monkeypatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# local credentials\n"
        "INFOBUDGET_ENV_TEST='from-file'\n"
        "export INFOBUDGET_EXISTING_TEST=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("INFOBUDGET_ENV_TEST", raising=False)
    monkeypatch.setenv("INFOBUDGET_EXISTING_TEST", "from-process")

    assert load_env_file(env_path) is True
    assert load_env_file(tmp_path / "missing.env") is False
    assert os.environ["INFOBUDGET_ENV_TEST"] == "from-file"
    assert os.environ["INFOBUDGET_EXISTING_TEST"] == "from-process"

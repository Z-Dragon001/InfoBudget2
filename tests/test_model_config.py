"""Tests for model tier configuration."""

from __future__ import annotations

from infobudget.config import load_project_bundle
from infobudget.schemas import ModelSpec


def test_model_tiers_use_selected_qwen_models() -> None:
    bundle = load_project_bundle("configs")

    small = bundle.models["small"]
    medium = bundle.models["medium"]
    large = bundle.models["large"]

    assert small.deploy == "local"
    assert small.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert small.api_base_url == "http://localhost:8001/v1"
    assert small.dtype == "float16"

    assert medium.deploy == "local"
    assert medium.model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    assert medium.api_base_url == "http://localhost:8002/v1"
    assert medium.dtype == "fp8"

    assert large.deploy == "api"
    assert large.model_name == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert large.api_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert large.api_key
    assert large.effective_model_name == "qwen3-next-80b-a3b-instruct"

    judge_model = bundle.config.evaluation.judge_model
    assert judge_model is not None
    assert judge_model.deploy == "api"
    assert judge_model.model_name == "gpt-4o-mini"
    assert judge_model.api_base_url == "https://api.openai.com/v1"
    assert judge_model.effective_model_name == "gpt-4o-mini"
    assert bundle.config.extractor.extraction_mode == "flat"


def test_model_spec_resolves_api_key_from_environment(monkeypatch) -> None:
    spec = ModelSpec(
        deploy="api",
        backend="openai_compatible",
        model_name="example-model",
        tokenizer_name="example-model",
        max_context_tokens=4096,
        tensor_parallel_size=1,
        dtype="n/a",
        api_key="${DASHSCOPE_API_KEY}",
    )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    assert spec.resolved_api_key() == "test-key"

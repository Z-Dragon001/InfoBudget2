"""Optional integration test for DashScope Qwen large-tier availability.

Run explicitly with:
RUN_DASHSCOPE_INTEGRATION_TESTS=1 DASHSCOPE_API_KEY=... pytest tests/test_dashscope_qwen_integration.py -q
"""

from __future__ import annotations

import os

import pytest

from infobudget.config import load_project_bundle
from infobudget.extractors.llm_joint import OpenAICompatibleClient


@pytest.mark.skipif(
    os.getenv("RUN_DASHSCOPE_INTEGRATION_TESTS") != "1",
    reason="set RUN_DASHSCOPE_INTEGRATION_TESTS=1 to call DashScope",
)
def test_dashscope_qwen3_next_80b_a3b_instruct_is_available() -> None:
    """Verify the configured Alibaba Cloud Qwen large model can answer a chat request."""
    bundle = load_project_bundle("configs")
    model_spec = bundle.models["large"]

    assert model_spec.deploy == "api"
    assert model_spec.backend == "openai_compatible"
    assert model_spec.model_name == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert model_spec.effective_model_name == "qwen3-next-80b-a3b-instruct"
    assert model_spec.api_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not model_spec.resolved_api_key():
        pytest.skip("DASHSCOPE_API_KEY is not set")

    response = OpenAICompatibleClient(timeout_seconds=60).complete(
        model_spec=model_spec,
        prompt="Reply with one short English sentence confirming that the model is available.",
        max_new_tokens=48,
        json_mode=False,
    )

    assert response.content.strip()
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.latency_ms > 0

from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import pytest

from infobudget.rl_router.api import LLMResponse, ModelAPIError
from infobudget.schemas import ModelSpec
from reference_fact_pipeline.cloudflare_api import (
    CloudflareResponsesClient,
    CloudflareWholesaleRateLimitError,
    ReferenceAPIClient,
    cloudflare_responses_endpoint,
    parse_cloudflare_responses_response,
    require_reference_api_credentials,
)


def _cloudflare_model() -> ModelSpec:
    return ModelSpec(
        deploy="api",
        backend="cloudflare_responses",
        model_name="openai/gpt-5.6-luna",
        request_model_name="openai/gpt-5.6-luna",
        tokenizer_name="gpt-5.6-luna",
        max_context_tokens=1_050_000,
        max_output_tokens=128_000,
        tensor_parallel_size=1,
        dtype="n/a",
        api_base_url=(
            "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        ),
        api_key_env="TEST_CLOUDFLARE_TOKEN",
    )


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload
        self.status = 200
        self.headers = {"x-request-id": "cf-request-1"}

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_cloudflare_responses_client_uses_documented_endpoint_and_payload(monkeypatch):
    monkeypatch.setenv("TEST_CLOUDFLARE_TOKEN", "secret-token")
    monkeypatch.setenv("TEST_CLOUDFLARE_ACCOUNT", "account123")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "id": "resp-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"facts":[]}',
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 9,
                    "total_tokens": 129,
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = CloudflareResponsesClient(
        max_retries=0, account_id_env="TEST_CLOUDFLARE_ACCOUNT"
    ).complete(
        model_spec=_cloudflare_model(),
        prompt="Return JSON only.",
        max_new_tokens=4096,
        json_mode=True,
    )

    assert captured["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/account123/ai/v1/responses"
    )
    assert captured["payload"] == {
        "model": "openai/gpt-5.6-luna",
        "input": "Return JSON only.",
        "max_output_tokens": 4096,
        "store": False,
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "text"}, "verbosity": "low"},
    }
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert result.content == '{"facts":[]}'
    assert (result.input_tokens, result.output_tokens) == (120, 9)
    assert result.provider_request_id == "cf-request-1"
    assert result.finish_reason == "completed"


def test_cloudflare_parser_supports_v4_envelope_and_rejects_failed_status():
    normalized = parse_cloudflare_responses_response(
        {
            "success": True,
            "result": {
                "id": "resp-2",
                "status": "completed",
                "output_text": "ok",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    )
    assert normalized["output_text"] == "ok"
    with pytest.raises(ModelAPIError, match="status=failed"):
        parse_cloudflare_responses_response(
            {"status": "failed", "error": {"message": "bad request"}}
        )


def test_cloudflare_preflight_requires_token_and_account(monkeypatch):
    model = _cloudflare_model()
    monkeypatch.setenv("TEST_CLOUDFLARE_TOKEN", "present")
    monkeypatch.delenv("TEST_CLOUDFLARE_ACCOUNT", raising=False)
    with pytest.raises(RuntimeError) as error:
        require_reference_api_credentials(
            {"gold": model},
            ["gold"],
            operation="test",
            account_id_env="TEST_CLOUDFLARE_ACCOUNT",
        )
    message = str(error.value)
    assert "TEST_CLOUDFLARE_ACCOUNT" in message
    monkeypatch.setenv("TEST_CLOUDFLARE_ACCOUNT", "account123")
    monkeypatch.delenv("TEST_CLOUDFLARE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TEST_CLOUDFLARE_TOKEN"):
        require_reference_api_credentials(
            {"gold": model},
            ["gold"],
            operation="test",
            account_id_env="TEST_CLOUDFLARE_ACCOUNT",
        )


def test_backend_router_keeps_chat_and_responses_transports_separate():
    calls = []

    class FakeClient:
        def __init__(self, name):
            self.name = name

        def complete(self, **kwargs):
            calls.append(self.name)
            return LLMResponse("ok", 1, 1, 1)

    router = ReferenceAPIClient(FakeClient("chat"), FakeClient("responses"))
    router.complete(
        model_spec=_cloudflare_model(), prompt="x", max_new_tokens=1, json_mode=True
    )
    chat_model = _cloudflare_model()
    chat_model.backend = "openai_compatible"
    router.complete(
        model_spec=chat_model, prompt="x", max_new_tokens=1, json_mode=True
    )
    assert calls == ["responses", "chat"]


def test_cloudflare_endpoint_rejects_unresolved_or_invalid_account_id():
    with pytest.raises(ModelAPIError, match="invalid Cloudflare account ID"):
        cloudflare_responses_endpoint(_cloudflare_model(), account_id="bad/account")


def _wholesale_402() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/responses",
        402,
        "Payment Required",
        {},
        BytesIO(
            b'{"error":{"code":"invalid_prompt","message":"Wholesale rate limit exceeded for this gateway. Please reduce request rate or use BYOK."}}'
        ),
    )


def test_wholesale_402_uses_long_backoff_then_recovers(monkeypatch):
    monkeypatch.setenv("TEST_CLOUDFLARE_TOKEN", "secret-token")
    monkeypatch.setenv("TEST_CLOUDFLARE_ACCOUNT", "account123")
    calls = 0
    sleeps = []

    def fake_urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise _wholesale_402()
        return _FakeResponse(
            {
                "id": "resp-recovered",
                "status": "completed",
                "output_text": '{"facts":[]}',
                "usage": {"input_tokens": 10, "output_tokens": 3},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("reference_fact_pipeline.cloudflare_api.time.sleep", sleeps.append)
    result = CloudflareResponsesClient(
        max_retries=0,
        account_id_env="TEST_CLOUDFLARE_ACCOUNT",
        wholesale_402_backoff_seconds=(60.0, 120.0, 300.0),
    ).complete(
        model_spec=_cloudflare_model(),
        prompt="Return JSON only.",
        max_new_tokens=100,
        json_mode=True,
    )
    assert result.content == '{"facts":[]}'
    assert result.retry_count == 3
    assert sleeps == [60.0, 120.0, 300.0]


def test_persistent_wholesale_402_opens_campaign_circuit(monkeypatch):
    monkeypatch.setenv("TEST_CLOUDFLARE_TOKEN", "secret-token")
    monkeypatch.setenv("TEST_CLOUDFLARE_ACCOUNT", "account123")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(_wholesale_402()))
    monkeypatch.setattr("reference_fact_pipeline.cloudflare_api.time.sleep", lambda _delay: None)
    client = CloudflareResponsesClient(
        max_retries=0,
        account_id_env="TEST_CLOUDFLARE_ACCOUNT",
        wholesale_402_backoff_seconds=(60.0, 120.0, 300.0),
    )
    with pytest.raises(CloudflareWholesaleRateLimitError) as error:
        client.complete(
            model_spec=_cloudflare_model(),
            prompt="Return JSON only.",
            max_new_tokens=100,
            json_mode=True,
        )
    assert len(error.value.attempts) == 4


def test_cloudflare_request_interval_is_enforced(monkeypatch):
    clock = iter((100.0, 100.0, 101.0, 103.0))
    sleeps = []
    monkeypatch.setattr(
        "reference_fact_pipeline.cloudflare_api.time.monotonic", lambda: next(clock)
    )
    monkeypatch.setattr("reference_fact_pipeline.cloudflare_api.time.sleep", sleeps.append)
    client = CloudflareResponsesClient(request_interval_seconds=3.0)
    client._wait_for_request_slot()
    client._wait_for_request_slot()
    assert sleeps == [2.0]

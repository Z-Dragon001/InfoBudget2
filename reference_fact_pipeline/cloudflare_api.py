"""Cloudflare Responses transport used only by frozen-reference Fact construction."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Mapping

from infobudget.rl_router.api import (
    ChatCompletionClient,
    LLMResponse,
    ModelAPIError,
    OpenAICompatibleClient,
    require_api_keys,
)
from infobudget.schemas import ModelSpec
from infobudget.utils.text import count_tokens

CLOUDFLARE_RESPONSES_BACKENDS = {
    "cloudflare_responses",
    "cloudflare_ai_responses",
}


def require_reference_api_credentials(
    model_specs: Mapping[str, ModelSpec],
    roles: Iterable[str],
    *,
    operation: str,
    account_id_env: str = "CLOUDFLARE_ACCOUNT_ID",
) -> None:
    """Run common API-key checks plus the Gold-only Cloudflare Account ID check."""
    selected = tuple(roles)
    require_api_keys(model_specs, selected, operation=operation)
    needs_cloudflare = any(
        model_specs[role].backend.strip().lower() in CLOUDFLARE_RESPONSES_BACKENDS
        for role in selected
    )
    if needs_cloudflare and not os.getenv(account_id_env, "").strip():
        raise RuntimeError(
            f"missing Cloudflare account ID environment variable for {operation}: "
            f"{account_id_env}"
        )


@dataclass(slots=True)
class CloudflareResponsesClient:
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    account_id_env: str = "CLOUDFLARE_ACCOUNT_ID"

    def complete(
        self,
        *,
        model_spec: ModelSpec,
        prompt: str,
        max_new_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        del json_mode  # Gold prompts already require JSON-only output.
        api_key = model_spec.resolved_api_key()
        if not api_key:
            raise ModelAPIError(
                f"missing API key environment variable {model_spec.api_key_env}",
                retryable=False,
            )
        account_id = os.getenv(self.account_id_env, "").strip()
        if not account_id:
            raise ModelAPIError(
                f"missing Cloudflare account ID environment variable {self.account_id_env}",
                retryable=False,
            )
        endpoint = cloudflare_responses_endpoint(model_spec, account_id=account_id)
        payload = {
            "model": model_spec.effective_model_name,
            "input": prompt,
            "max_output_tokens": max_new_tokens,
            "store": False,
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "text"}, "verbosity": "low"},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "InfoBudget-GoldFact/0.1",
            "Authorization": f"Bearer {api_key}",
        }
        started = time.perf_counter()
        parsed, attempts, request_id = self._post(
            endpoint,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers,
        )
        try:
            response = parse_cloudflare_responses_response(parsed)
        except ModelAPIError as exc:
            if not exc.attempts:
                exc.attempts = attempts
            raise
        content = response["output_text"]
        usage = response.get("usage")
        provider_usage = _has_complete_usage(usage)
        return LLMResponse(
            content=content,
            input_tokens=_usage_int(usage, "input_tokens", count_tokens(prompt)),
            output_tokens=_usage_int(usage, "output_tokens", count_tokens(content)),
            latency_ms=max(1, int((time.perf_counter() - started) * 1000)),
            usage_source="provider" if provider_usage else "tokenizer_estimate",
            retry_count=max(0, len(attempts) - 1),
            provider_request_id=request_id or str(response.get("id") or ""),
            finish_reason=str(response.get("status") or ""),
            attempts=attempts,
        )

    def _post(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        attempts: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            started = time.perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    request_id = _request_id(response.headers)
                    attempts.append(
                        {
                            "transport_attempt": attempt + 1,
                            "started_at": started_at,
                            "status": "succeeded",
                            "http_status": int(getattr(response, "status", 200)),
                            "latency_ms": max(
                                1, int((time.perf_counter() - started) * 1000)
                            ),
                            "provider_request_id": request_id,
                            "cost_status": "reported_on_logical_response",
                        }
                    )
                    try:
                        parsed = json.loads(raw)
                        if not isinstance(parsed, dict):
                            raise TypeError("response root must be an object")
                    except (json.JSONDecodeError, TypeError) as exc:
                        attempts[-1].update(
                            status="failed",
                            error=f"invalid provider response: {exc}"[:500],
                            cost_status="unknown",
                        )
                        raise ModelAPIError(
                            f"Cloudflare returned invalid JSON: {exc}",
                            attempts=attempts,
                        ) from exc
                    return parsed, attempts, request_id
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").strip()
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                attempts.append(
                    _failed_attempt(
                        attempt + 1,
                        started_at,
                        started,
                        http_status=int(exc.code),
                        request_id=_request_id(exc.headers),
                        retryable=retryable,
                        error=body or str(exc),
                    )
                )
                if not retryable or attempt >= self.max_retries:
                    raise ModelAPIError(
                        f"Cloudflare request failed: HTTP {exc.code}: {body[:500]}",
                        attempts=attempts,
                        retryable=retryable,
                    ) from exc
                last_error = exc
                self._sleep(attempt, exc.headers)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                attempts.append(
                    _failed_attempt(
                        attempt + 1,
                        started_at,
                        started,
                        http_status=None,
                        request_id="",
                        retryable=True,
                        error=str(exc),
                    )
                )
                if attempt >= self.max_retries:
                    raise ModelAPIError(
                        f"Cloudflare request failed: {exc}", attempts=attempts
                    ) from exc
                last_error = exc
                self._sleep(attempt, None)
        raise ModelAPIError(
            f"Cloudflare request failed: {last_error}", attempts=attempts
        )

    def _sleep(self, attempt: int, headers: Any) -> None:
        delay = _retry_after_seconds(headers)
        if delay is None:
            base = self.retry_backoff_seconds * (2**attempt)
            delay = base + random.uniform(0.0, max(0.0, base * 0.25))
        time.sleep(max(0.0, delay))


@dataclass(slots=True)
class ReferenceAPIClient:
    """Gold-only dispatcher; no candidate, QA, or Judge call site uses this class."""

    chat_client: ChatCompletionClient
    cloudflare_client: ChatCompletionClient

    def complete(self, *, model_spec: ModelSpec, **kwargs: Any) -> LLMResponse:
        client = (
            self.cloudflare_client
            if model_spec.backend.strip().lower() in CLOUDFLARE_RESPONSES_BACKENDS
            else self.chat_client
        )
        return client.complete(model_spec=model_spec, **kwargs)


def build_reference_api_client(
    *, timeout_seconds: int, max_retries: int, retry_backoff_seconds: float
) -> ReferenceAPIClient:
    options = {
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
    }
    return ReferenceAPIClient(
        chat_client=OpenAICompatibleClient(**options),
        cloudflare_client=CloudflareResponsesClient(**options),
    )


def cloudflare_responses_endpoint(model_spec: ModelSpec, *, account_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", account_id):
        raise ModelAPIError("invalid Cloudflare account ID", retryable=False)
    base = model_spec.api_base_url.strip() or (
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
    )
    base = base.replace("{account_id}", account_id)
    if "{" in base or "}" in base:
        raise ModelAPIError(
            "unresolved placeholder in Cloudflare api_base_url", retryable=False
        )
    base = base.rstrip("/")
    return base if base.endswith("/responses") else base + "/responses"


def parse_cloudflare_responses_response(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("success") is False:
        raise ModelAPIError(
            f"Cloudflare API error: {payload.get('errors') or payload}", retryable=False
        )
    value = payload.get("result", payload)
    if not isinstance(value, dict):
        raise ModelAPIError("Cloudflare response result must be an object")
    status = str(value.get("status") or "")
    if status and status != "completed":
        raise ModelAPIError(
            f"Cloudflare response status={status}: "
            f"{value.get('error') or value.get('incomplete_details')}",
            retryable=status == "in_progress",
        )
    direct = value.get("output_text")
    parts = [direct] if isinstance(direct, str) and direct else []
    if not parts:
        for item in value.get("output", ()):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", ()):
                if (
                    isinstance(content, dict)
                    and content.get("type") == "output_text"
                    and isinstance(content.get("text"), str)
                ):
                    parts.append(content["text"])
    output_text = "".join(parts).strip()
    if not output_text:
        raise ModelAPIError("Cloudflare response contains no output_text")
    normalized = dict(value)
    normalized["output_text"] = output_text
    return normalized


def _usage_int(usage: Any, key: str, default: int) -> int:
    if isinstance(usage, dict):
        try:
            return int(usage[key])
        except (KeyError, TypeError, ValueError):
            pass
    return int(default)


def _has_complete_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    try:
        return int(usage["input_tokens"]) >= 0 and int(usage["output_tokens"]) >= 0
    except (KeyError, TypeError, ValueError):
        return False


def _request_id(headers: Any) -> str:
    if headers is None:
        return ""
    for key in ("x-request-id", "request-id", "x-trace-id", "trace-id"):
        try:
            value = headers.get(key)
        except AttributeError:
            value = None
        if value:
            return str(value)
    return ""


def _failed_attempt(
    attempt: int,
    started_at: str,
    started: float,
    *,
    http_status: int | None,
    request_id: str,
    retryable: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "transport_attempt": attempt,
        "started_at": started_at,
        "status": "failed",
        "http_status": http_status,
        "latency_ms": max(1, int((time.perf_counter() - started) * 1000)),
        "provider_request_id": request_id,
        "retryable": retryable,
        "error": error[:500],
        "cost_status": "unknown",
    }


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return min(300.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return min(
                300.0,
                max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds()),
            )
        except (TypeError, ValueError, OverflowError):
            return None

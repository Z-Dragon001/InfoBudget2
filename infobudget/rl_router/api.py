"""OpenAI-compatible API client used by fact extraction, reader, and judge roles."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Mapping, Protocol

from infobudget.schemas import ModelSpec
from infobudget.utils.text import count_tokens


class ModelAPIError(RuntimeError):
    """A model failure carrying every transport attempt for later cost auditing."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, Any]] | None = None,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.attempts = list(attempts or [])
        self.retryable = bool(retryable)


@dataclass(slots=True)
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    usage_source: str = "provider"
    retry_count: int = 0
    provider_request_id: str = ""
    finish_reason: str = ""
    attempts: list[dict[str, Any]] | None = None


class ChatCompletionClient(Protocol):
    def complete(
        self,
        *,
        model_spec: ModelSpec,
        prompt: str,
        max_new_tokens: int,
        json_mode: bool,
    ) -> LLMResponse: ...


def require_api_keys(
    model_specs: Mapping[str, ModelSpec],
    roles: Iterable[str],
    *,
    operation: str,
) -> None:
    """Fail before creating run artifacts when selected API-backed roles lack credentials."""
    missing: list[str] = []
    for role in roles:
        if role not in model_specs:
            raise ValueError(f"unknown model role: {role}")
        spec = model_specs[role]
        if spec.deploy == "api" and not spec.resolved_api_key():
            missing.append(f"{role}={spec.api_key_env or '<unconfigured>'}")
    if missing:
        raise RuntimeError(
            f"missing API key environment variables for {operation}: {', '.join(missing)}"
        )


@dataclass(slots=True)
class OpenAICompatibleClient:
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    def complete(
        self,
        *,
        model_spec: ModelSpec,
        prompt: str,
        max_new_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        if not model_spec.api_base_url:
            raise ModelAPIError(
                f"missing api_base_url for model {model_spec.model_name}", retryable=False
            )
        api_key = model_spec.resolved_api_key()
        if model_spec.deploy == "api" and not api_key:
            raise ModelAPIError(
                f"missing API key environment variable {model_spec.api_key_env}",
                retryable=False,
            )
        payload: dict[str, Any] = {
            "model": model_spec.effective_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": max_new_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "InfoBudget/0.1",
            "Authorization": f"Bearer {api_key}",
        }
        started = time.perf_counter()
        try:
            parsed, attempts, provider_request_id = self._post(
                model_spec.api_base_url.rstrip("/") + "/chat/completions",
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers,
            )
        except ModelAPIError:
            raise
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError(
                "response does not match chat-completions schema", attempts=attempts
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelAPIError("response content is empty", attempts=attempts)
        usage = parsed.get("usage")
        provider_usage = _has_complete_provider_usage(usage)
        return LLMResponse(
            content=content,
            input_tokens=_usage_int(usage, ("prompt_tokens", "input_tokens"), count_tokens(prompt)),
            output_tokens=_usage_int(usage, ("completion_tokens", "output_tokens"), count_tokens(content)),
            latency_ms=latency_ms,
            usage_source="provider" if provider_usage else "tokenizer_estimate",
            retry_count=max(0, len(attempts) - 1),
            provider_request_id=provider_request_id,
            finish_reason=str(parsed["choices"][0].get("finish_reason") or ""),
            attempts=attempts,
        )

    def _post(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        last_error: BaseException | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            attempt_started = time.perf_counter()
            attempt_started_at = datetime.now(timezone.utc).isoformat()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    provider_request_id = _request_id(response.headers)
                    attempts.append(
                        {
                            "transport_attempt": attempt + 1,
                            "started_at": attempt_started_at,
                            "status": "succeeded",
                            "http_status": int(getattr(response, "status", 200)),
                            "latency_ms": max(1, int((time.perf_counter() - attempt_started) * 1000)),
                            "provider_request_id": provider_request_id,
                            "cost_status": "reported_on_logical_response",
                        }
                    )
                    try:
                        parsed = parse_chat_completion_response(raw)
                    except (json.JSONDecodeError, ModelAPIError, TypeError, ValueError) as exc:
                        attempts[-1]["status"] = "failed"
                        attempts[-1]["error"] = f"invalid provider response: {exc}"[:500]
                        attempts[-1]["cost_status"] = "unknown"
                        raise ModelAPIError(
                            f"provider returned an invalid chat response: {exc}",
                            attempts=attempts,
                        ) from exc
                    return parsed, attempts, provider_request_id
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").strip()
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                attempts.append(
                    {
                        "transport_attempt": attempt + 1,
                        "started_at": attempt_started_at,
                        "status": "failed",
                        "http_status": int(exc.code),
                        "latency_ms": max(1, int((time.perf_counter() - attempt_started) * 1000)),
                        "provider_request_id": _request_id(exc.headers),
                        "retryable": retryable,
                        "error": body[:500] or str(exc),
                        "cost_status": "unknown",
                    }
                )
                if not retryable or attempt >= self.max_retries:
                    raise ModelAPIError(
                        f"model request failed: HTTP {exc.code}: {body[:500]}",
                        attempts=attempts,
                        retryable=retryable,
                    ) from exc
                last_error = exc
                self._sleep_before_retry(attempt, exc.headers)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                attempts.append(
                    {
                        "transport_attempt": attempt + 1,
                        "started_at": attempt_started_at,
                        "status": "failed",
                        "http_status": None,
                        "latency_ms": max(1, int((time.perf_counter() - attempt_started) * 1000)),
                        "provider_request_id": "",
                        "retryable": True,
                        "error": str(exc)[:500],
                        "cost_status": "unknown",
                    }
                )
                if attempt >= self.max_retries:
                    raise ModelAPIError(
                        f"model request failed: {exc}", attempts=attempts
                    ) from exc
                last_error = exc
                self._sleep_before_retry(attempt, None)
        raise ModelAPIError(f"model request failed: {last_error}", attempts=attempts)

    def _sleep_before_retry(self, attempt: int, headers: Any) -> None:
        retry_after = _retry_after_seconds(headers)
        if retry_after is None:
            base = self.retry_backoff_seconds * (2**attempt)
            retry_after = base + random.uniform(0.0, max(0.0, base * 0.25))
        time.sleep(max(0.0, retry_after))


def parse_chat_completion_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text.startswith("data:"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ModelAPIError("chat response root must be an object")
        return parsed
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    model = ""
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        chunk = json.loads(payload)
        model = str(chunk.get("model") or model)
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
    return {"model": model, "choices": [{"message": {"content": "".join(content_parts)}}], "usage": usage}


def _usage_int(usage: Any, keys: tuple[str, ...], default: int) -> int:
    if isinstance(usage, dict):
        for key in keys:
            value = usage.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return int(default)


def _has_complete_provider_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    for keys in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        found = False
        for key in keys:
            if key not in usage:
                continue
            try:
                if int(usage[key]) < 0:
                    return False
            except (TypeError, ValueError):
                return False
            found = True
            break
        if not found:
            return False
    return True


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

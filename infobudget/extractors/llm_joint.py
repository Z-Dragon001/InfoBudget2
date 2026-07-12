"""OpenAI-compatible joint memory extractors."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from infobudget.cost.logger import CostLogger
from infobudget.extractors.base import JointMemoryExtractor
from infobudget.extractors.mock_joint import MockJointExtractor
from infobudget.runtime.model_registry import ModelRegistry
from infobudget.schemas import MemoryEntry, ModelSpec, ScoreResult, Segment, Tier
from infobudget.utils.text import count_tokens


class LLMExtractionError(RuntimeError):
    """Raised when an LLM extraction call or response cannot be used."""


@dataclass(slots=True)
class LLMResponse:
    """Normalized chat-completion response."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class ChatCompletionClient(Protocol):
    """Protocol for OpenAI-compatible chat completion clients."""

    def complete(
        self,
        *,
        model_spec: ModelSpec,
        prompt: str,
        max_new_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """Return a single assistant message."""


@dataclass(slots=True)
class OpenAICompatibleClient:
    """Small standard-library client for OpenAI-compatible chat completions."""

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
            raise LLMExtractionError(f"missing api_base_url for model {model_spec.model_name}")
        api_key = model_spec.resolved_api_key()
        if model_spec.deploy == "api" and not api_key:
            raise LLMExtractionError(f"missing API key for model {model_spec.model_name}")
        url = model_spec.api_base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": model_spec.effective_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": max_new_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OpenAI/Python 1.0.0",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        start = time.perf_counter()
        raw = self._post_with_retries(url, data, headers, model_spec.model_name)

        latency_ms = max(1, int((time.perf_counter() - start) * 1000))
        try:
            parsed = parse_chat_completion_response(raw)
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMExtractionError("LLM response does not match chat-completions schema") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMExtractionError("LLM response content is empty")

        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        input_tokens = _int_from_usage(usage, "prompt_tokens", count_tokens(prompt))
        output_tokens = _int_from_usage(usage, "completion_tokens", count_tokens(content))
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def _post_with_retries(self, url: str, data: bytes, headers: dict[str, str], model_name: str) -> str:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").strip()
                detail = f"HTTP {exc.code}: {body[:500]}" if body else f"HTTP {exc.code}"
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise LLMExtractionError(f"LLM request failed for {model_name}: {detail}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt >= self.max_retries:
                    raise LLMExtractionError(f"LLM request failed for {model_name}: {exc}") from exc
                last_error = exc
            time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise LLMExtractionError(f"LLM request failed for {model_name}: {last_error}")


def parse_chat_completion_response(raw: str) -> dict[str, Any]:
    """Parse either ordinary OpenAI JSON or SSE chat-completion chunks."""
    text = raw.strip()
    if not text.startswith("data:"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise TypeError("chat completion response must be a JSON object")
        return parsed

    chunks: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        chunk = json.loads(payload)
        if isinstance(chunk, dict):
            chunks.append(chunk)
    if not chunks:
        raise ValueError("empty SSE chat completion response")

    content_parts: list[str] = []
    role = "assistant"
    finish_reason = None
    usage = None
    for chunk in chunks:
        if chunk.get("usage") is not None:
            usage = chunk.get("usage")
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        if isinstance(delta, dict):
            role = delta.get("role") or role
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
        if isinstance(message, dict):
            role = message.get("role") or role
            if isinstance(message.get("content"), str):
                content_parts.append(message["content"])

    merged = dict(chunks[-1])
    merged["choices"] = [
        {
            "index": 0,
            "message": {"role": role, "content": "".join(content_parts)},
            "finish_reason": finish_reason,
        }
    ]
    if usage is not None:
        merged["usage"] = usage
    return merged


@dataclass(slots=True)
class OpenAICompatibleJointExtractor(JointMemoryExtractor):
    """Joint memory extractor backed by an OpenAI-compatible endpoint."""

    model_registry: ModelRegistry
    cost_logger: CostLogger
    prompt_template: str | dict[str, str]
    relational_prompt_template: str | dict[str, str] | None = None
    max_new_tokens: int = 1024
    json_mode: bool = True
    client: ChatCompletionClient | None = None
    extractor_name: str = "openai_compatible_joint_extractor"
    extraction_mode: str = "flat"

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OpenAICompatibleClient()

    def prompt_for_tier(self, tier: Tier) -> str:
        """Return the prompt template associated with the routed tier."""
        return self._prompt_for_tier(self.prompt_template, tier)

    def relational_prompt_for_tier(self, tier: Tier) -> str:
        """Return the relational prompt template associated with the routed tier."""
        return self._prompt_for_tier(self.relational_prompt_template or self.prompt_template, tier)

    @staticmethod
    def _prompt_for_tier(prompt_template: str | dict[str, str], tier: Tier) -> str:
        if isinstance(prompt_template, dict):
            return prompt_template.get(tier) or prompt_template.get("default") or next(iter(prompt_template.values()))
        return prompt_template

    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> list[MemoryEntry]:
        mode = self.extraction_mode.strip().lower()
        if mode not in {"flat", "event"}:
            raise LLMExtractionError(f"unsupported extraction_mode: {self.extraction_mode}")
        factual_entries = self._extract_once(
            segment,
            tier,
            score_result,
            self.prompt_for_tier(tier),
            log_mode="flat_factual" if mode == "flat" else "event_factual",
            default_entry_type="factual",
        )
        if mode == "flat":
            return factual_entries
        relational_entries = self._extract_once(
            segment,
            tier,
            score_result,
            self.relational_prompt_for_tier(tier),
            log_mode="event_relational",
            default_entry_type="relational",
        )
        return factual_entries + relational_entries

    def _extract_once(
        self,
        segment: Segment,
        tier: Tier,
        score_result: ScoreResult,
        prompt_template: str,
        *,
        log_mode: str,
        default_entry_type: str,
    ) -> list[MemoryEntry]:
        model_spec = self.model_registry.get(tier)
        prompt = render_prompt(
            prompt_template,
            tier=tier,
            information_score=score_result.final_score,
            segment_text=segment.text,
        )
        if self.client is None:
            raise LLMExtractionError("missing chat completion client")
        response = self.client.complete(
            model_spec=model_spec,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            json_mode=self.json_mode,
        )
        payload = parse_memory_json(response.content)
        self.cost_logger.log_extraction(
            segment_id=segment.segment_id,
            tier=tier,
            model_spec=model_spec,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            extraction_mode=log_mode,
        )
        return memory_entries_from_payload(
            payload,
            segment=segment,
            default_entry_type=default_entry_type,
        )


class LocalJointExtractor(OpenAICompatibleJointExtractor):
    """Joint extractor for local OpenAI-compatible inference servers."""


class APIJointExtractor(OpenAICompatibleJointExtractor):
    """Joint extractor for remote OpenAI-compatible APIs."""


@dataclass(slots=True)
class TieredJointExtractor(JointMemoryExtractor):
    """Delegate each routed tier to the local or API extractor configured for its model."""

    model_registry: ModelRegistry
    local_extractor: LocalJointExtractor
    api_extractor: APIJointExtractor
    fallback_extractor: MockJointExtractor | None = None
    fallback_on_error: bool = True

    def prompt_for_tier(self, tier: Tier) -> str:
        """Return the prompt template associated with the routed tier."""
        return self._extractor_for_tier(tier).prompt_for_tier(tier)

    def extract(self, segment: Segment, tier: Tier, score_result: ScoreResult) -> list[MemoryEntry]:
        extractor = self._extractor_for_tier(tier)
        try:
            return extractor.extract(segment, tier, score_result)
        except LLMExtractionError:
            if not self.fallback_on_error or self.fallback_extractor is None:
                raise
            return self.fallback_extractor.extract(segment, tier, score_result)

    def _extractor_for_tier(self, tier: Tier) -> OpenAICompatibleJointExtractor:
        model_spec = self.model_registry.get(tier)
        if model_spec.deploy == "api":
            return self.api_extractor
        return self.local_extractor


def render_prompt(template: str, *, tier: Tier, information_score: float, segment_text: str) -> str:
    """Render known placeholders without treating JSON braces as format slots."""
    escaped = (
        template.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{router_level}}", "{router_level}")
        .replace("{{information_score}}", "{information_score}")
        .replace("{{segment_text}}", "{segment_text}")
    )
    return escaped.format(
        router_level=tier,
        information_score=f"{information_score:.4f}",
        segment_text=segment_text,
    )


def parse_memory_json(content: str) -> dict[str, Any]:
    """Parse a model response into a memory JSON object."""
    text = _strip_json_fence(content.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMExtractionError("LLM response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMExtractionError("LLM response JSON must be an object")
    return parsed


def memory_entries_from_payload(
    payload: dict[str, Any],
    *,
    segment: Segment,
    default_entry_type: str = "factual",
) -> list[MemoryEntry]:
    """Map LightMem-style LLM JSON into atomic memory entries."""
    rows = _list(payload.get("data"))
    source_map = _source_metadata_by_id(segment)
    topic_id = _topic_id_from_segment(segment)
    entries: list[MemoryEntry] = []
    for row in rows:
        item = _dict(row)
        memory_text = _string(item.get("fact")) or _string(item.get("relation"))
        if not memory_text:
            continue
        entry_type = _string(item.get("entry_type")) or ("relational" if _string(item.get("relation")) else default_entry_type)
        source_id = _int(item.get("source_id"), segment.start_turn - 1)
        source = source_map.get(source_id) or source_map.get(segment.start_turn - 1) or {}
        time_stamp = source.get("time_stamp", "")
        source_turn_id = source_id + 1 if source_id >= 0 else 0
        entries.append(
            MemoryEntry(
                time_stamp=time_stamp,
                float_time_stamp=_float_timestamp(time_stamp),
                weekday=source.get("weekday", ""),
                topic_id=topic_id,
                topic_summary="",
                memory=memory_text,
                original_memory=memory_text,
                compressed_memory="",
                entry_type=entry_type,
                speaker_id=source.get("speaker_id", "unknown"),
                speaker_name=source.get("speaker_name", "User"),
                consolidated=False,
                update_queue=[],
                source_segment_id=segment.segment_id,
                source_turn_id=source_turn_id,
                source_turn_ids=list(segment.turn_ids),
                source_start_turn=segment.start_turn,
                source_end_turn=segment.end_turn,
            )
        )
    return entries


def _strip_json_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


_SOURCE_LINE_RE = re.compile(
    r"^\[(?P<time>[^\],]+)(?:,\s*(?P<weekday>[^\]]+))?\]\s*"
    r"(?P<source_id>\d+)\.(?P<speaker>[^:]+):"
)


def _source_metadata_by_id(segment: Segment) -> dict[int, dict[str, str]]:
    metadata: dict[int, dict[str, str]] = {}
    for line in segment.text.splitlines():
        match = _SOURCE_LINE_RE.match(line.strip())
        if not match:
            continue
        source_id = int(match.group("source_id"))
        speaker = match.group("speaker").strip()
        metadata[source_id] = {
            "time_stamp": match.group("time").strip(),
            "weekday": (match.group("weekday") or "").strip(),
            "speaker_id": speaker.lower() or "unknown",
            "speaker_name": speaker or "User",
        }
    return metadata


def _topic_id_from_segment(segment: Segment) -> int:
    match = re.search(r"(\d+)$", segment.segment_id)
    if match:
        return max(0, int(match.group(1)) - 1)
    return max(0, segment.start_turn - 1)


def _float_timestamp(time_stamp: str) -> float:
    if not time_stamp:
        return 0.0
    text = time_stamp.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _int_from_usage(usage: Any, key: str, default: int) -> int:
    if isinstance(usage, dict):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

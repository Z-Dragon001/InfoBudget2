"""LightMEM-compatible QA retrieval, answer generation, and LLM judging over S."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable, Protocol

from infobudget.rl_router.api import ChatCompletionClient, OpenAICompatibleClient
from infobudget.rl_router.config import RLConfigBundle
from infobudget.rl_router.embedding import Encoder
from infobudget.rl_router.ledger import SqliteLedger
from infobudget.rl_router.qdrant_store import FactQdrantStore


@dataclass(slots=True)
class ReaderResult:
    answer: str
    prompt_name: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    retry_count: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class JudgeResult:
    correct: bool
    prompt_name: str
    model_name: str
    raw_response: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    retry_count: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class QAEvaluation:
    question_id: str
    predicted_answer: str
    correct: bool
    retrieved: list[dict]
    reader_prompt_name: str = ""
    judge_prompt_name: str = ""
    reader_model_name: str = ""
    judge_model_name: str = ""
    reader_input_tokens: int = 0
    reader_output_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    reader_input_cost: float = 0.0
    reader_output_cost: float = 0.0
    judge_input_cost: float = 0.0
    judge_output_cost: float = 0.0
    reader_latency_ms: int = 0
    judge_latency_ms: int = 0
    judge_raw_response: str = ""
    reader_retry_count: int = 0
    judge_retry_count: int = 0
    reader_attempts: list[dict[str, Any]] = field(default_factory=list)
    judge_attempts: list[dict[str, Any]] = field(default_factory=list)


class QAReader(Protocol):
    def answer(
        self,
        *,
        dataset_name: str,
        question: dict,
        retrieved: list[dict],
        sample_metadata: dict[str, Any],
    ) -> ReaderResult: ...


class QAJudge(Protocol):
    def judge(self, *, dataset_name: str, question: dict, predicted_answer: str) -> JudgeResult: ...


@dataclass(slots=True)
class LightMemQAReader:
    """Render the unmodified LightMEM answer prompts and call the fixed QA model."""

    bundle: RLConfigBundle
    client: ChatCompletionClient = field(default_factory=OpenAICompatibleClient)

    def answer(
        self,
        *,
        dataset_name: str,
        question: dict,
        retrieved: list[dict],
        sample_metadata: dict[str, Any],
    ) -> ReaderResult:
        normalized = dataset_name.strip().lower()
        if normalized == "locomo":
            prompt_name = "locomo_answer"
            speaker_1, speaker_2 = _locomo_speakers(question, sample_metadata, retrieved)
            first, second = _group_locomo_memories(retrieved, speaker_1, speaker_2)
            values = {
                "speaker_1_name": speaker_1,
                "speaker_1_memories": first,
                "speaker_2_name": speaker_2,
                "speaker_2_memories": second,
                "question": str(question.get("question") or ""),
            }
        elif normalized == "longmemeval":
            prompt_name = "longmemeval_answer"
            values = {
                "question_date": str(question.get("question_date") or sample_metadata.get("question_date") or ""),
                "question": str(question.get("question") or ""),
                "memories": _render_memory_lines(retrieved),
            }
        else:
            raise ValueError(f"unsupported QA dataset: {dataset_name!r}")

        prompt = self.bundle.prompt_path(prompt_name).read_text(encoding="utf-8").format(**values)
        model = self.bundle.project.models["qa_reader"]
        response = self.client.complete(
            model_spec=model,
            prompt=prompt,
            max_new_tokens=int(self.bundle.rl["evaluation"]["reader_max_new_tokens"]),
            json_mode=False,
        )
        input_cost, output_cost = _call_costs(self.bundle, model.model_name, response.input_tokens, response.output_tokens)
        return ReaderResult(
            answer=response.content.strip(),
            prompt_name=prompt_name,
            model_name=model.effective_model_name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            input_cost=input_cost,
            output_cost=output_cost,
            retry_count=response.retry_count,
            attempts=response.attempts or [],
        )


@dataclass(slots=True)
class LightMemLLMJudge:
    """Route to the exact LightMEM judge prompt for each dataset/question type."""

    bundle: RLConfigBundle
    client: ChatCompletionClient = field(default_factory=OpenAICompatibleClient)

    def judge(self, *, dataset_name: str, question: dict, predicted_answer: str) -> JudgeResult:
        prompt_name = select_judge_prompt(
            dataset_name,
            str(question.get("question_type") or ""),
            judge_profile=str(question.get("judge_profile") or ""),
            is_unanswerable=bool(question.get("is_unanswerable")),
            question_id=str(question.get("question_id") or ""),
        )
        template = Template(self.bundle.prompt_path(prompt_name).read_text(encoding="utf-8"))
        prompt = template.substitute(
            question=str(question.get("question") or ""),
            golden_answers=str(question.get("answer") or question.get("gold_answer") or ""),
            prediction=predicted_answer,
        )
        model = self.bundle.project.models["judge_llm"]
        is_locomo = dataset_name.strip().lower() == "locomo"
        response = self.client.complete(
            model_spec=model,
            prompt=prompt,
            max_new_tokens=int(self.bundle.rl["evaluation"]["judge_max_new_tokens"]),
            json_mode=is_locomo,
        )
        correct = parse_locomo_judge_label(response.content) if is_locomo else parse_longmemeval_judge_label(response.content)
        input_cost, output_cost = _call_costs(self.bundle, model.model_name, response.input_tokens, response.output_tokens)
        return JudgeResult(
            correct=correct,
            prompt_name=prompt_name,
            model_name=model.effective_model_name,
            raw_response=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            input_cost=input_cost,
            output_cost=output_cost,
            retry_count=response.retry_count,
            attempts=response.attempts or [],
        )


class AssemblyEvaluator:
    """Retrieve only from one physical S assembly, then run Reader and Judge."""

    def __init__(
        self,
        *,
        store: FactQdrantStore,
        encoder: Encoder,
        reader: QAReader,
        judge: QAJudge,
        top_k: int = 20,
        ledger_path: str | Path | None = None,
    ):
        self.store, self.encoder, self.reader, self.judge, self.top_k = store, encoder, reader, judge, top_k
        if ledger_path:
            path = Path(ledger_path)
            legacy = path if path.suffix.lower() == ".jsonl" else None
            database = path.with_suffix(".sqlite3") if legacy else path
            self.ledger = SqliteLedger(
                database,
                "evaluations",
                ("assembly_id", "question_id"),
                legacy_jsonl_path=legacy,
            )
        else:
            self.ledger = None

    def evaluate_question(
        self,
        question: dict,
        *,
        dataset_name: str,
        split: str,
        sample_id: str,
        assembly_id: str,
        sample_metadata: dict[str, Any] | None = None,
    ) -> QAEvaluation:
        query = self.encoder.encode([str(question["question"])])[0]
        hits = self.store.search_assembly(
            query,
            dataset_name=dataset_name,
            split=split,
            sample_id=sample_id,
            assembly_id=assembly_id,
            top_k=self.top_k,
        )
        retrieved = [{**payload, "retrieval_score": score} for payload, score in hits]
        reader_result = self.reader.answer(
            dataset_name=dataset_name,
            question=question,
            retrieved=retrieved,
            sample_metadata=sample_metadata or {},
        )
        judge_result = self.judge.judge(
            dataset_name=dataset_name,
            question=question,
            predicted_answer=reader_result.answer,
        )
        result = QAEvaluation(
            question_id=str(question.get("question_id") or ""),
            predicted_answer=reader_result.answer,
            correct=judge_result.correct,
            retrieved=retrieved,
            reader_prompt_name=reader_result.prompt_name,
            judge_prompt_name=judge_result.prompt_name,
            reader_model_name=reader_result.model_name,
            judge_model_name=judge_result.model_name,
            reader_input_tokens=reader_result.input_tokens,
            reader_output_tokens=reader_result.output_tokens,
            judge_input_tokens=judge_result.input_tokens,
            judge_output_tokens=judge_result.output_tokens,
            reader_input_cost=reader_result.input_cost,
            reader_output_cost=reader_result.output_cost,
            judge_input_cost=judge_result.input_cost,
            judge_output_cost=judge_result.output_cost,
            reader_latency_ms=reader_result.latency_ms,
            judge_latency_ms=judge_result.latency_ms,
            judge_raw_response=judge_result.raw_response,
            reader_retry_count=reader_result.retry_count,
            judge_retry_count=judge_result.retry_count,
            reader_attempts=reader_result.attempts,
            judge_attempts=judge_result.attempts,
        )
        if self.ledger is not None:
            self.ledger.append(
                {
                    "assembly_id": assembly_id,
                    "dataset_name": dataset_name,
                    "split": split,
                    "sample_id": sample_id,
                    "category": str(question.get("category") or ""),
                    "question_type": str(question.get("question_type") or ""),
                    "judge_profile": str(question.get("judge_profile") or "generic"),
                    "is_unanswerable": bool(question.get("is_unanswerable")),
                    **asdict(result),
                }
            )
        return result

    def evaluate_sample(
        self,
        questions: list[dict],
        *,
        sample_metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[QAEvaluation], None] | None = None,
        **scope: str,
    ) -> tuple[float, list[QAEvaluation]]:
        results = []
        for question in questions:
            result = self.evaluate_question(
                question, sample_metadata=sample_metadata, **scope
            )
            results.append(result)
            if progress_callback is not None:
                progress_callback(result)
        return (sum(item.correct for item in results) / len(results) if results else 0.0), results


def build_lightmem_evaluator(
    bundle: RLConfigBundle,
    *,
    store: FactQdrantStore,
    encoder: Encoder,
    client: ChatCompletionClient | None = None,
    ledger_path: str | Path | None = None,
) -> AssemblyEvaluator:
    shared_client = client or OpenAICompatibleClient()
    return AssemblyEvaluator(
        store=store,
        encoder=encoder,
        reader=LightMemQAReader(bundle, shared_client),
        judge=LightMemLLMJudge(bundle, shared_client),
        top_k=int(bundle.rl["evaluation"]["retrieval_top_k"]),
        ledger_path=ledger_path,
    )


def select_judge_prompt(
    dataset_name: str,
    question_type: str,
    *,
    judge_profile: str = "",
    is_unanswerable: bool = False,
    question_id: str = "",
) -> str:
    normalized_dataset = dataset_name.strip().lower()
    if normalized_dataset == "locomo":
        return "locomo_judge"
    if normalized_dataset != "longmemeval":
        raise ValueError(f"unsupported judge dataset: {dataset_name!r}")
    if is_unanswerable or "abs" in question_id.casefold() or judge_profile == "longmemeval_abstention":
        return "longmemeval_abstention_judge"
    normalized = question_type.strip().lower()
    mapping = {
        "single-session-user": "longmemeval_single_session_judge",
        "single-session-assistant": "longmemeval_single_session_judge",
        "multi-session": "longmemeval_single_session_judge",
        "temporal-reasoning": "longmemeval_temporal_reasoning_judge",
        "knowledge-update": "longmemeval_knowledge_update_judge",
        "single-session-preference": "longmemeval_preference_judge",
    }
    return mapping.get(normalized, "longmemeval_exact_match_judge")


def parse_locomo_judge_label(content: str) -> bool:
    text = content.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LoCoMo judge must return a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("LoCoMo judge response root must be a JSON object")
    label = str(payload.get("label") or "").strip().upper()
    if label == "CORRECT":
        return True
    if label == "WRONG":
        return False
    raise ValueError(f"invalid LoCoMo judge label: {label!r}")


def parse_longmemeval_judge_label(content: str) -> bool:
    first_line = content.strip().splitlines()[0].strip() if content.strip() else ""
    token = re.sub(r"[^a-z]", "", first_line.casefold())
    if token in {"yes", "correct"}:
        return True
    if token in {"no", "wrong", "incorrect"}:
        return False
    raise ValueError(f"invalid LongMemEval judge response: {first_line!r}")


def parse_judge_label(content: str) -> bool:
    """Backward-compatible parser for the old CORRECT/INCORRECT protocol."""
    first_line = content.strip().splitlines()[0].strip().upper() if content.strip() else ""
    if first_line == "CORRECT":
        return True
    if first_line == "INCORRECT":
        return False
    raise ValueError(f"invalid judge response: {first_line!r}")


def _locomo_speakers(
    question: dict,
    sample_metadata: dict[str, Any],
    retrieved: list[dict],
) -> tuple[str, str]:
    speaker_1 = str(sample_metadata.get("speaker_a") or question.get("speaker_a") or "").strip()
    speaker_2 = str(sample_metadata.get("speaker_b") or question.get("speaker_b") or "").strip()
    seen: list[str] = []
    for item in retrieved:
        name = str(item.get("source_speaker") or item.get("speaker_name") or "").strip()
        if name and name.casefold() not in {value.casefold() for value in seen}:
            seen.append(name)
    if not speaker_1 and seen:
        speaker_1 = seen[0]
    if not speaker_2:
        speaker_2 = next((name for name in seen if name.casefold() != speaker_1.casefold()), "")
    return speaker_1 or "User", speaker_2 or "Assistant"


def _group_locomo_memories(retrieved: list[dict], speaker_1: str, speaker_2: str) -> tuple[str, str]:
    first: list[str] = []
    second: list[str] = []
    for item in retrieved:
        source = str(item.get("source_speaker") or item.get("speaker_name") or "").casefold()
        fact = str(item.get("fact_text") or "")
        line = _render_memory_line(item)
        if source == speaker_2.casefold():
            second.append(line)
        elif source == speaker_1.casefold():
            first.append(line)
        elif _mentions(fact, speaker_2) and not _mentions(fact, speaker_1):
            second.append(line)
        else:
            first.append(line)
    return "\n".join(first) or "(none)", "\n".join(second) or "(none)"


def _mentions(text: str, name: str) -> bool:
    return bool(name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.I))


def _render_memory_lines(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(no retrieved memories)"
    return "\n".join(_render_memory_line(item) for item in retrieved)


def _render_memory_line(item: dict) -> str:
    timestamp = str(item.get("source_timestamp") or item.get("segment_start_timestamp") or "").strip()
    weekday = str(item.get("source_weekday") or "").strip()
    prefix = f"[{timestamp}, {weekday}]" if timestamp and weekday else (f"[{timestamp}]" if timestamp else "")
    fact = str(item.get("fact_text") or "").strip()
    return f"{prefix} {fact}".strip()


def _call_costs(bundle: RLConfigBundle, model_name: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
    price = bundle.project.prices[model_name]
    return (
        input_tokens * price.official_price_in_per_1m / 1_000_000,
        output_tokens * price.official_price_out_per_1m / 1_000_000,
    )

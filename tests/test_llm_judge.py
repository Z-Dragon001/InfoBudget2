"""Tests for the independent LLM evaluation judge."""

from __future__ import annotations

from infobudget.evaluation.judges import LLMJudge, JudgeRegistry
from infobudget.extractors.llm_joint import LLMResponse
from infobudget.schemas import DatasetQAPair, ModelSpec


class FakeJudgeClient:
    def __init__(self, content: str = '{"correct": true, "matched_by": "semantic_equivalence"}') -> None:
        self.calls: list[dict] = []
        self.content = content

    def complete(self, *, model_spec: ModelSpec, prompt: str, max_new_tokens: int, json_mode: bool) -> LLMResponse:
        self.calls.append(
            {
                "model_name": model_spec.model_name,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "json_mode": json_mode,
            }
        )
        return LLMResponse(
            content=self.content,
            input_tokens=32,
            output_tokens=8,
            latency_ms=10,
        )


def _judge_model() -> ModelSpec:
    return ModelSpec(
        deploy="api",
        backend="openai_compatible",
        model_name="gpt-4o-mini",
        tokenizer_name="gpt-4o-mini",
        api_base_url="https://api.openai.com/v1",
        api_key="${OPENAI_API_KEY}",
        api_key_env="OPENAI_API_KEY",
        request_model_name="gpt-4o-mini",
        max_context_tokens=128000,
        tensor_parallel_size=1,
        dtype="n/a",
    )


def test_llm_judge_uses_configured_model_without_cost_logger() -> None:
    client = FakeJudgeClient()
    judge = LLMJudge(_judge_model(), client=client)
    qa_pair = DatasetQAPair(
        question_id="q1",
        question="What does Alice prefer?",
        answer="Alice prefers tea.",
    )

    result = judge.judge(qa_pair, "She likes tea.", [])

    assert result.correct is True
    assert result.matched_by == "llm_judge:semantic_equivalence"
    assert client.calls[0]["model_name"] == "gpt-4o-mini"
    assert client.calls[0]["json_mode"] is True


def test_judge_registry_creates_llm_judge_when_configured() -> None:
    judge = JudgeRegistry.create(
        "generic",
        judge_mode="llm_judge",
        judge_model=_judge_model(),
        client=FakeJudgeClient(),
    )

    assert isinstance(judge, LLMJudge)


def test_locomo_llm_judge_uses_accuracy_prompt_json_label() -> None:
    client = FakeJudgeClient('{"label": "CORRECT"}')
    judge = LLMJudge(_judge_model(), client=client)
    qa_pair = DatasetQAPair(
        question_id="q1",
        question="What did Alice buy?",
        answer="A shell necklace",
        judge_profile="locomo_qa",
    )

    result = judge.judge(qa_pair, "She bought a shell necklace.", [])

    assert result.correct is True
    assert result.matched_by == "llm_judge:locomo_correct"
    assert client.calls[0]["json_mode"] is True
    assert "Gold answer: A shell necklace" in client.calls[0]["prompt"]


def test_longmemeval_llm_judge_uses_yes_no_prompt() -> None:
    client = FakeJudgeClient("yes")
    judge = LLMJudge(_judge_model(), client=client)
    qa_pair = DatasetQAPair(
        question_id="q1",
        question="How many days did it take?",
        answer="18 days",
        question_type="temporal-reasoning",
        judge_profile="longmemeval_temporal_reasoning",
    )

    result = judge.judge(qa_pair, "19 days", [])

    assert result.correct is True
    assert result.matched_by == "llm_judge:longmemeval_yes_no"
    assert client.calls[0]["json_mode"] is False
    assert "off-by-one errors" in client.calls[0]["prompt"]

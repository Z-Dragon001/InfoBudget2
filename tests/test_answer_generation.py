"""Tests for dataset-specific QA answer generation."""

from __future__ import annotations

from dataclasses import replace

from infobudget.config import load_project_bundle
from infobudget.evaluation.answer_generation import DatasetAnswerGenerator
from infobudget.extractors.llm_joint import LLMResponse
from infobudget.schemas import DatasetDialogueExample, DatasetQAPair, MemoryEntry, ModelSpec


class FakeAnswerClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, *, model_spec: ModelSpec, prompt: str, max_new_tokens: int, json_mode: bool) -> LLMResponse:
        self.calls.append(
            {
                "model_name": model_spec.model_name,
                "effective_model_name": model_spec.effective_model_name,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "json_mode": json_mode,
            }
        )
        return LLMResponse(content="Business Administration", input_tokens=50, output_tokens=3, latency_ms=12)


def test_llm_answer_generation_uses_medium_model_and_longmemeval_prompt() -> None:
    bundle = load_project_bundle("configs")
    bundle = type(bundle)(
        root_dir=bundle.root_dir,
        config=replace(
            bundle.config,
            evaluation=replace(bundle.config.evaluation, qa_mode="llm_qa", answer_model_tier="medium"),
        ),
        weights=bundle.weights,
        models=bundle.models,
        prices=bundle.prices,
        prompt_dir=bundle.prompt_dir,
    )
    client = FakeAnswerClient()
    generator = DatasetAnswerGenerator(bundle, client=client)
    example = DatasetDialogueExample(
        sample_id="longmem_1",
        dataset_name="longmemeval",
        split="full",
        sessions=[],
        dialogue=[],
        qa_pairs=[],
        metadata={},
    )
    qa_pair = DatasetQAPair(
        question_id="longmem_1",
        question="What degree did I graduate with?",
        answer="Business Administration",
        question_date="2023/05/30 (Tue) 23:40",
    )
    memories = [
        MemoryEntry(
            time_stamp="2023-05-20T02:21:00.000",
            weekday="Sat",
            speaker_name="User",
            memory="The user graduated with a Business Administration degree.",
        )
    ]

    result = generator.generate(
        dataset_name="longmemeval",
        example=example,
        qa_pair=qa_pair,
        retrieved_entries=memories,
    )

    assert result.answer == "Business Administration"
    assert result.answer_model_tier == "medium"
    assert client.calls[0]["model_name"] == bundle.models["medium"].model_name
    assert client.calls[0]["json_mode"] is False
    assert "Question time: 2023/05/30 (Tue) 23:40" in client.calls[0]["prompt"]
    assert "Business Administration degree" in client.calls[0]["prompt"]

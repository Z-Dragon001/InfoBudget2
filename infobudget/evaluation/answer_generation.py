"""LLM-backed answer generation for dataset memory evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from infobudget.config import ProjectBundle
from infobudget.extractors.llm_joint import ChatCompletionClient, LLMResponse, OpenAICompatibleClient
from infobudget.runtime.prompt_loader import load_prompt
from infobudget.schemas import DatasetDialogueExample, DatasetQAPair, MemoryEntry, ModelSpec


@dataclass(slots=True)
class AnswerResult:
    """Generated answer and lightweight model-call metadata."""

    answer: str
    matched_by: str
    model_name: str = ""
    answer_model_tier: str = ""
    prompt_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass(slots=True)
class DatasetAnswerGenerator:
    """Generate answers from retrieved memories with dataset-specific prompts."""

    bundle: ProjectBundle
    client: ChatCompletionClient | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OpenAICompatibleClient(timeout_seconds=120)

    def generate(
        self,
        *,
        dataset_name: str,
        example: DatasetDialogueExample,
        qa_pair: DatasetQAPair,
        retrieved_entries: list[MemoryEntry],
    ) -> AnswerResult:
        mode = self.bundle.config.evaluation.qa_mode.strip().lower()
        if mode in {"retrieved_top1", "top1_memory", "rule"}:
            return self._top1_memory_answer(qa_pair, retrieved_entries)
        if mode != "llm_qa":
            raise ValueError(f"unsupported evaluation.qa_mode: {self.bundle.config.evaluation.qa_mode}")
        return self._llm_answer(dataset_name, example, qa_pair, retrieved_entries)

    @staticmethod
    def _top1_memory_answer(qa_pair: DatasetQAPair, retrieved_entries: list[MemoryEntry]) -> AnswerResult:
        if not retrieved_entries:
            answer = "I don't know." if qa_pair.is_unanswerable else ""
            return AnswerResult(answer=answer, matched_by="no_retrieved_memory")
        return AnswerResult(answer=retrieved_entries[0].memory, matched_by="retrieved_top1")

    def _llm_answer(
        self,
        dataset_name: str,
        example: DatasetDialogueExample,
        qa_pair: DatasetQAPair,
        retrieved_entries: list[MemoryEntry],
    ) -> AnswerResult:
        model_spec = self._answer_model()
        prompt_name, prompt = self._build_prompt(dataset_name, example, qa_pair, retrieved_entries)
        if self.client is None:
            raise RuntimeError("missing chat completion client for answer generation")
        response = self.client.complete(
            model_spec=model_spec,
            prompt=prompt,
            max_new_tokens=self.bundle.config.evaluation.qa_max_new_tokens,
            json_mode=False,
        )
        return self._from_response(response, model_spec, prompt_name)

    def _answer_model(self) -> ModelSpec:
        tier = self.bundle.config.evaluation.answer_model_tier
        if tier not in self.bundle.models:
            raise KeyError(f"evaluation.answer_model_tier must be one of {sorted(self.bundle.models)}; got {tier}")
        return self.bundle.models[tier]

    def _build_prompt(
        self,
        dataset_name: str,
        example: DatasetDialogueExample,
        qa_pair: DatasetQAPair,
        retrieved_entries: list[MemoryEntry],
    ) -> tuple[str, str]:
        normalized = dataset_name.strip().lower()
        if normalized == "locomo":
            template = load_prompt(self.bundle.prompt_dir, "locomo_answer.txt")
            speaker_1, speaker_2 = _locomo_speakers(example, retrieved_entries)
            speaker_memories = _group_memories_by_speaker(retrieved_entries, speaker_1, speaker_2)
            return (
                "locomo_answer",
                template.format(
                    speaker_1_name=speaker_1,
                    speaker_1_memories=speaker_memories[0],
                    speaker_2_name=speaker_2,
                    speaker_2_memories=speaker_memories[1],
                    question=qa_pair.question,
                ),
            )
        if normalized == "longmemeval":
            template = load_prompt(self.bundle.prompt_dir, "longmemeval_answer.txt")
            return (
                "longmemeval_answer",
                template.format(
                    question_date=qa_pair.question_date or example.metadata.get("question_date", ""),
                    question=qa_pair.question,
                    memories=_render_memory_lines(retrieved_entries),
                ),
            )
        template = load_prompt(self.bundle.prompt_dir, "longmemeval_answer.txt")
        return (
            "generic_answer",
            template.format(
                question_date=qa_pair.question_date or "",
                question=qa_pair.question,
                memories=_render_memory_lines(retrieved_entries),
            ),
        )

    def _from_response(self, response: LLMResponse, model_spec: ModelSpec, prompt_name: str) -> AnswerResult:
        return AnswerResult(
            answer=response.content.strip(),
            matched_by="llm_qa",
            model_name=model_spec.effective_model_name,
            answer_model_tier=self.bundle.config.evaluation.answer_model_tier,
            prompt_name=prompt_name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


def _locomo_speakers(example: DatasetDialogueExample, retrieved_entries: list[MemoryEntry]) -> tuple[str, str]:
    speaker_1 = str(example.metadata.get("speaker_a") or "").strip()
    speaker_2 = str(example.metadata.get("speaker_b") or "").strip()
    if speaker_1 and speaker_2:
        return speaker_1, speaker_2
    seen: list[str] = []
    for entry in retrieved_entries:
        name = entry.speaker_name.strip()
        if name and name.casefold() not in {item.casefold() for item in seen}:
            seen.append(name)
    while len(seen) < 2:
        seen.append("User" if not seen else "Assistant")
    return speaker_1 or seen[0], speaker_2 or seen[1]


def _group_memories_by_speaker(retrieved_entries: list[MemoryEntry], speaker_1: str, speaker_2: str) -> tuple[str, str]:
    speaker_1_lines: list[str] = []
    speaker_2_lines: list[str] = []
    speaker_1_key = speaker_1.casefold()
    speaker_2_key = speaker_2.casefold()
    for entry in retrieved_entries:
        name = entry.speaker_name.casefold()
        line = _render_memory_line(entry)
        if name == speaker_2_key:
            speaker_2_lines.append(line)
        elif name == speaker_1_key or not speaker_1_lines:
            speaker_1_lines.append(line)
        else:
            speaker_2_lines.append(line)
    return "\n".join(speaker_1_lines) or "(none)", "\n".join(speaker_2_lines) or "(none)"


def _render_memory_lines(retrieved_entries: list[MemoryEntry]) -> str:
    if not retrieved_entries:
        return "(no retrieved memories)"
    return "\n".join(_render_memory_line(entry) for entry in retrieved_entries)


def _render_memory_line(entry: MemoryEntry) -> str:
    timestamp = entry.time_stamp
    if timestamp and entry.weekday:
        prefix = f"[{timestamp}, {entry.weekday}]"
    elif timestamp:
        prefix = f"[{timestamp}]"
    else:
        prefix = ""
    speaker = f"{entry.speaker_name}: " if entry.speaker_name else ""
    return f"{prefix} {speaker}{entry.memory}".strip()

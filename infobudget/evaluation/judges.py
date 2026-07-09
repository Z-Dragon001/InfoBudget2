"""功能：实现数据集与问题类型可扩展的评估判定器。
输入：问答样本、预测答案与检索上下文。
输出：是否正确与判定元数据。
依赖：dataclasses、schemas、text。
作者：OpenAI Codex
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from infobudget.extractors.llm_joint import ChatCompletionClient, LLMResponse, OpenAICompatibleClient
from infobudget.schemas import DatasetQAPair, MemoryEntry, ModelSpec
from infobudget.utils.text import content_tokens


@dataclass(slots=True)
class JudgeResult:
    """评估判定结果。"""

    correct: bool
    matched_by: str


class BaseJudge:
    """判定器基类。"""

    def judge(self, qa_pair: DatasetQAPair, predicted_answer: str, retrieved_entries: list[MemoryEntry]) -> JudgeResult:
        raise NotImplementedError


class GenericSubstringJudge(BaseJudge):
    """通用子串 / token overlap 判定器。"""

    def judge(self, qa_pair: DatasetQAPair, predicted_answer: str, retrieved_entries: list[MemoryEntry]) -> JudgeResult:
        gold_answer = qa_pair.answer
        if not gold_answer:
            return JudgeResult(False, "empty_gold")
        if qa_pair.is_unanswerable:
            return self._judge_abstention(predicted_answer)
        gold = gold_answer.casefold()
        predicted = predicted_answer.casefold()
        if gold in predicted:
            return JudgeResult(True, "predicted_substring")
        contexts = self._build_context(retrieved_entries)
        if gold in contexts:
            return JudgeResult(True, "retrieved_context")
        if self._token_overlap(gold_answer, contexts) >= 0.5:
            return JudgeResult(True, "token_overlap")
        return JudgeResult(False, "no_match")

    @staticmethod
    def _judge_abstention(predicted_answer: str) -> JudgeResult:
        answer = predicted_answer.casefold()
        abstain_markers = [
            "don't know",
            "do not know",
            "cannot answer",
            "can't answer",
            "insufficient",
            "not enough information",
            "unknown",
            "无法回答",
            "不知道",
            "信息不足",
        ]
        matched = any(marker in answer for marker in abstain_markers)
        return JudgeResult(matched, "abstention_marker" if matched else "abstention_miss")

    @staticmethod
    def _build_context(retrieved_entries: list[MemoryEntry]) -> str:
        return " ".join(entry.memory for entry in retrieved_entries).casefold()

    @staticmethod
    def _token_overlap(gold_answer: str, contexts: str) -> float:
        gold_tokens = set(content_tokens(gold_answer))
        if not gold_tokens:
            return 0.0
        context_tokens = set(content_tokens(contexts))
        return len(gold_tokens & context_tokens) / len(gold_tokens)


class LongMemEvalTemporalJudge(GenericSubstringJudge):
    """LongMemEval temporal-reasoning 专用判定器。"""

    def judge(self, qa_pair: DatasetQAPair, predicted_answer: str, retrieved_entries: list[MemoryEntry]) -> JudgeResult:
        base = super().judge(qa_pair, predicted_answer, retrieved_entries)
        if base.correct:
            return base
        gold = qa_pair.answer.casefold()
        predicted = predicted_answer.casefold()
        gold_numbers = self._extract_numbers(gold)
        predicted_numbers = self._extract_numbers(predicted)
        if len(gold_numbers) == 1 and len(predicted_numbers) == 1:
            if abs(gold_numbers[0] - predicted_numbers[0]) <= 1:
                return JudgeResult(True, "off_by_one_temporal")
        return base

    @staticmethod
    def _extract_numbers(text: str) -> list[int]:
        numbers: list[int] = []
        current = ""
        for ch in text:
            if ch.isdigit():
                current += ch
            elif current:
                numbers.append(int(current))
                current = ""
        if current:
            numbers.append(int(current))
        return numbers


@dataclass(slots=True)
class LLMJudge(BaseJudge):
    """OpenAI-compatible LLM judge used only for evaluation."""

    model_spec: ModelSpec
    client: ChatCompletionClient | None = None
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OpenAICompatibleClient(timeout_seconds=60)

    def judge(self, qa_pair: DatasetQAPair, predicted_answer: str, retrieved_entries: list[MemoryEntry]) -> JudgeResult:
        if self.client is None:
            raise RuntimeError("missing chat completion client for LLM judge")
        prompt = self._build_prompt(qa_pair, predicted_answer, retrieved_entries)
        response = self.client.complete(
            model_spec=self.model_spec,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            json_mode=self._json_mode(qa_pair),
        )
        return self._parse_response(response, qa_pair)

    @staticmethod
    def _build_prompt(qa_pair: DatasetQAPair, predicted_answer: str, retrieved_entries: list[MemoryEntry]) -> str:
        if qa_pair.judge_profile == "locomo_qa":
            return LOCOMO_ACCURACY_PROMPT.format(
                question=qa_pair.question,
                gold_answer=qa_pair.answer,
                generated_answer=predicted_answer,
            )
        if qa_pair.judge_profile.startswith("longmemeval_"):
            return get_longmemeval_anscheck_prompt(
                qa_pair.question_type,
                qa_pair.question,
                qa_pair.answer,
                predicted_answer,
                abstention=qa_pair.is_unanswerable or qa_pair.judge_profile == "longmemeval_abstention",
            )
        context = _render_retrieved_context(retrieved_entries)
        unanswerable_policy = (
            "The gold answer marks this question as unanswerable. The prediction is correct only if it clearly "
            "abstains or says the information is insufficient."
            if qa_pair.is_unanswerable
            else "The question is answerable. Accept paraphrases and semantically equivalent answers."
        )
        return f"""You are an impartial evaluation judge for long-term memory question answering.
Decide whether the predicted answer is correct relative to the gold answer and the retrieved memory context.
{unanswerable_policy}
Do not require exact string matching. Penalize contradictions, unsupported extra claims, and missing key facts.

Question:
{qa_pair.question}

Gold answer:
{qa_pair.answer}

Predicted answer:
{predicted_answer}

Retrieved memory context:
{context}

Return JSON only with this schema:
{{"correct": true, "matched_by": "short reason"}}"""

    @staticmethod
    def _json_mode(qa_pair: DatasetQAPair) -> bool:
        return not qa_pair.judge_profile.startswith("longmemeval_")

    @staticmethod
    def _parse_response(response: LLMResponse, qa_pair: DatasetQAPair) -> JudgeResult:
        if qa_pair.judge_profile == "locomo_qa":
            payload = _parse_json_object(response.content)
            label = _string(payload.get("label")).casefold()
            correct = label in {"correct", "yes", "true", "1"}
            return JudgeResult(correct, f"llm_judge:locomo_{label or 'empty_label'}")
        if qa_pair.judge_profile.startswith("longmemeval_"):
            correct = true_or_false(response.content)
            return JudgeResult(correct, "llm_judge:longmemeval_yes_no")
        payload = _parse_json_object(response.content)
        correct = _bool(payload.get("correct"))
        matched_by = _string(payload.get("matched_by")) or "llm_judge"
        return JudgeResult(correct, f"llm_judge:{matched_by}")


class JudgeRegistry:
    """判定器注册表。"""

    _JUDGES = {
        "generic": GenericSubstringJudge,
        "locomo_qa": GenericSubstringJudge,
        "longmemeval_generic": GenericSubstringJudge,
        "longmemeval_single_session": GenericSubstringJudge,
        "longmemeval_knowledge_update": GenericSubstringJudge,
        "longmemeval_preference": GenericSubstringJudge,
        "longmemeval_abstention": GenericSubstringJudge,
        "longmemeval_temporal_reasoning": LongMemEvalTemporalJudge,
    }

    @classmethod
    def create(
        cls,
        judge_profile: str,
        judge_mode: str = "rule_judge",
        judge_model: ModelSpec | None = None,
        client: ChatCompletionClient | None = None,
    ) -> BaseJudge:
        if judge_mode == "llm_judge":
            if judge_model is None:
                raise ValueError("evaluation.judge_model is required when judge_mode is llm_judge")
            return LLMJudge(judge_model, client=client)
        return cls._JUDGES.get(judge_profile, GenericSubstringJudge)()


LOCOMO_ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def get_longmemeval_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool = False,
) -> str:
    """Return the LongMemEval judge prompt from the LightMem evaluation script."""
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. The model "
            "could say that the information is incomplete, or some other information is given but the asked "
            "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model "
            "correctly identify the question as unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)

    normalized = task.strip().lower()
    if normalized in {"single-session-user", "single-session-assistant", "multi-session"}:
        template = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
            "the correct answer or contains all the intermediate steps to get the correct answer, you should "
            "also answer yes. If the response only contains a subset of the information required by the answer, "
            "answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
            "correct? Answer yes or no only."
        )
    elif normalized == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
            "the correct answer or contains all the intermediate steps to get the correct answer, you should "
            "also answer yes. If the response only contains a subset of the information required by the answer, "
            "answer no. In addition, do not penalize off-by-one errors for the number of days. If the question "
            "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
            "predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: "
            "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    elif normalized == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response should be considered as correct "
            "as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    elif normalized == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized response, and a response from a "
            "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    else:
        template = (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no.\n\nQuestion: {}\n\nCorrect Answer: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
    return template.format(question, answer, response)


def true_or_false(response: str | None) -> bool:
    """Parse the first-line yes/no style response used by LongMemEval."""
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace(".", "").replace("!", "").replace(":", "").replace(";", "").split()
    if not tokens:
        return False
    head = tokens[0]
    if head in {"yes", "y"}:
        return True
    if head in {"no", "n"}:
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


def _render_retrieved_context(retrieved_entries: list[MemoryEntry]) -> str:
    if not retrieved_entries:
        return "(no retrieved memory)"
    blocks: list[str] = []
    for index, entry in enumerate(retrieved_entries, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Memory {index}]",
                    f"time={entry.time_stamp} {entry.weekday}".strip(),
                    f"memory={entry.memory}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM judge response must be a JSON object")
    return parsed


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "correct", "1"}
    return bool(value)


def _string(value) -> str:
    return value.strip() if isinstance(value, str) else ""

"""功能：提供轻量文本处理与启发式抽取工具。
输入：中英文混合文本。
输出：token、句子、实体与事件候选。
依赖：re、math。
作者：OpenAI Codex
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./+-]+|[\u4e00-\u9fff]")
LATIN_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{1,}\b")
CLAUSE_SPLIT_PATTERN = re.compile(r"[。！？!?;；,，:：.\n]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?;\n]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "we",
    "with",
    "了",
    "个",
    "们",
    "在",
    "是",
    "有",
    "和",
    "就",
    "也",
    "都",
    "我",
    "把",
    "的",
}

VERB_HINTS = {
    "build",
    "define",
    "design",
    "extract",
    "implement",
    "plan",
    "run",
    "study",
    "use",
    "做",
    "写",
    "实现",
    "开发",
    "构建",
    "测试",
    "研究",
    "设计",
}


PREDICATE_HINTS = VERB_HINTS | {
    "add",
    "analyze",
    "call",
    "change",
    "choose",
    "compare",
    "compute",
    "control",
    "create",
    "decide",
    "detect",
    "evaluate",
    "filter",
    "generate",
    "include",
    "measure",
    "need",
    "output",
    "record",
    "route",
    "save",
    "score",
    "select",
    "store",
    "update",
    "want",
}

SYNONYM_ALIASES = {
    "ai": "artificial_intelligence",
    "avg": "average",
    "bert": "language_model",
    "chatgpt": "language_model",
    "costs": "cost",
    "gpt": "language_model",
    "llm": "language_model",
    "llms": "language_model",
    "model": "language_model",
    "models": "language_model",
    "qa": "question_answering",
    "score": "scoring",
    "scores": "scoring",
    "thresholds": "threshold",
}

IRREGULAR_LEMMAS = {
    "built": "build",
    "ran": "run",
    "running": "run",
    "stored": "store",
    "using": "use",
    "used": "use",
}


def clamp01(value: float) -> float:
    """裁剪到 [0, 1]。"""
    return max(0.0, min(1.0, value))


def tokenize_text(text: str) -> list[str]:
    """执行轻量 token 切分。"""
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def count_tokens(text: str) -> int:
    """估计 token 数量。"""
    return len(tokenize_text(text))


def split_sentences(text: str) -> list[str]:
    """执行简单句切分。"""
    parts = [item.strip() for item in SENTENCE_SPLIT_PATTERN.split(text or "")]
    return [item for item in parts if item]


def content_tokens(text: str) -> list[str]:
    """筛选内容 token。"""
    tokens = tokenize_text(text)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 0]


def normalize_concept_token(token: str) -> str:
    """Normalize a candidate concept token before idea-unit extraction."""
    lowered = token.lower().strip("._+-/")
    if not lowered:
        return ""
    if lowered in IRREGULAR_LEMMAS:
        lowered = IRREGULAR_LEMMAS[lowered]
    elif re.fullmatch(r"[a-z][a-z0-9_./+-]*", lowered):
        if len(lowered) > 5 and lowered.endswith("ies"):
            lowered = lowered[:-3] + "y"
        elif len(lowered) > 5 and lowered.endswith("ing"):
            lowered = lowered[:-3]
            if len(lowered) > 2 and lowered[-1] == lowered[-2]:
                lowered = lowered[:-1]
        elif len(lowered) > 4 and lowered.endswith("ied"):
            lowered = lowered[:-3] + "y"
        elif len(lowered) > 4 and lowered.endswith("ed"):
            lowered = lowered[:-2]
        elif len(lowered) > 4 and lowered.endswith(("ches", "shes", "sses", "xes", "zes")):
            lowered = lowered[:-2]
        elif len(lowered) > 3 and lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
            lowered = lowered[:-1]
    return SYNONYM_ALIASES.get(lowered, lowered)


def is_content_pos_candidate(token: str) -> bool:
    """Heuristic POS filter for noun/verb/adjective/adverb/proper-noun candidates."""
    if token in STOPWORDS:
        return False
    if re.fullmatch(r"[a-z]", token):
        return False
    return bool(re.search(r"[a-z0-9_\u4e00-\u9fff]", token))


def normalized_content_terms(text: str) -> list[str]:
    """Run tokenization, POS-like filtering, normalization, stopword removal, and alias folding."""
    terms: list[str] = []
    for token in tokenize_text(text):
        if not is_content_pos_candidate(token):
            continue
        normalized = normalize_concept_token(token)
        if normalized and normalized not in STOPWORDS:
            terms.append(normalized)
    return terms


@lru_cache(maxsize=4)
def _load_spacy_model(model_name: str):
    try:
        import spacy  # type: ignore
    except ImportError:
        return None
    try:
        return spacy.load(model_name)
    except OSError:
        return None


def _spacy_term(token) -> str:
    raw = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
    return normalize_concept_token(raw)


def _extract_spacy_idea_units(text: str, model_name: str) -> list[str]:
    nlp = _load_spacy_model(model_name)
    if nlp is None:
        return []
    units: set[str] = set()
    doc = nlp(text or "")
    content_pos = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "NUM"}
    for sent in doc.sents:
        for token in sent:
            if token.is_stop or token.pos_ not in {"VERB", "AUX"}:
                continue
            predicate = _spacy_term(token)
            if not predicate:
                continue
            subjects = [
                _spacy_term(child)
                for child in token.children
                if child.dep_ in {"nsubj", "nsubjpass", "csubj"} and not child.is_stop
            ] or ["implicit"]
            objects = [
                _spacy_term(child)
                for child in token.children
                if child.dep_ in {"attr", "dative", "dobj", "obj", "oprd", "pobj"} and not child.is_stop
            ] or ["implicit"]
            for subject in subjects:
                for obj in objects:
                    if subject and obj and obj != predicate:
                        units.add(f"{subject}::{predicate}::{obj}")
        for token in sent:
            if token.is_stop or token.pos_ not in content_pos:
                continue
            head = token.head
            if head == token or head.is_stop or head.pos_ not in content_pos:
                continue
            left = _spacy_term(head)
            right = _spacy_term(token)
            relation = normalize_concept_token(token.dep_) or "rel"
            if left and right and left != right:
                units.add(f"{left}::{relation}::{right}")
    return sorted(units)


def extract_idea_units(text: str, spacy_model: str | None = None) -> list[str]:
    """Extract deduplicated proposition-like idea units for idea density."""
    if spacy_model:
        spacy_units = _extract_spacy_idea_units(text, spacy_model)
        if spacy_units:
            return spacy_units
    units: set[str] = set()
    for clause in CLAUSE_SPLIT_PATTERN.split(text or ""):
        terms = normalized_content_terms(clause)
        if not terms:
            continue
        predicate_indexes = [
            index
            for index, term in enumerate(terms)
            if term in PREDICATE_HINTS or term.endswith(("ize", "ise"))
        ]
        if predicate_indexes:
            for index in predicate_indexes:
                predicate = terms[index]
                subjects = terms[max(0, index - 2) : index] or ["implicit"]
                objects = terms[index + 1 : index + 4] or ["implicit"]
                subject = subjects[-1]
                for obj in objects:
                    if obj != predicate:
                        units.add(f"{subject}::{predicate}::{obj}")
            continue
        if len(terms) == 1:
            units.add(f"{terms[0]}::exists")
            continue
        for left, right in zip(terms, terms[1:]):
            if left != right:
                units.add(f"{left}::rel::{right}")
    return sorted(units)


def normalized_entropy(tokens: list[str]) -> float:
    """计算归一化熵。"""
    if len(tokens) <= 1:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    max_entropy = math.log2(len(counts))
    return clamp01(entropy / max_entropy) if max_entropy > 0 else 0.0


def extract_entities(text: str) -> list[str]:
    """启发式抽取实体名称。"""
    entities: list[str] = []
    seen: set[str] = set()
    for match in LATIN_ENTITY_PATTERN.findall(text or ""):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            entities.append(match)
    for token in tokenize_text(text):
        if any(ch.isdigit() for ch in token) and len(token) > 1 and token not in seen:
            seen.add(token)
            entities.append(token)
    return entities


def summarize_text(text: str, max_chars: int = 96) -> str:
    """生成检索友好的简短摘要。"""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def detect_topic(text: str) -> str:
    """根据高频词猜测主题。"""
    tokens = content_tokens(text)
    if not tokens:
        return "general"
    freq: dict[str, int] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return ranked[0][0]


def extract_preferences(text: str) -> list[tuple[str, str]]:
    """抽取简单偏好。"""
    prefs: list[tuple[str, str]] = []
    for sentence in split_sentences(text):
        if "prefer" in sentence.lower() or "喜欢" in sentence or "希望" in sentence:
            prefs.append(("user", summarize_text(sentence, 48)))
    return prefs


def extract_constraints(text: str) -> list[str]:
    """抽取简单约束。"""
    constraints: list[str] = []
    for sentence in split_sentences(text):
        if any(key in sentence for key in ["必须", "不要", "不得", "only", "must", "should not"]):
            constraints.append(summarize_text(sentence, 72))
    return constraints


def extract_event_clauses(text: str) -> list[str]:
    """抽取事件候选。"""
    events: list[str] = []
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(hint in lowered for hint in VERB_HINTS) or any(hint in sentence for hint in VERB_HINTS):
            events.append(summarize_text(sentence, 88))
    if not events and text.strip():
        events.append(summarize_text(text, 88))
    return events

"""功能：提供轻量文本处理与启发式抽取工具。
输入：中英文混合文本。
输出：token、句子、实体与事件候选。
依赖：re、math。
作者：OpenAI Codex
"""

from __future__ import annotations

import math
import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./+-]+|[\u4e00-\u9fff]")
LATIN_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{1,}\b")
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

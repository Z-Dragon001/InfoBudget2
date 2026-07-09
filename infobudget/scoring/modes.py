"""Scoring mode names and routing-score selection helpers."""

from __future__ import annotations

from typing import Literal

from infobudget.utils.text import clamp01

ScoringMode = Literal[
    "entropy_only",
    "lexical_density_only",
    "entity_density_only",
    "concept_density_only",
    "information_gain_only",
    "actionability_only",
    "intrinsic_only",
    "utility_only",
    "full",
]

SCORING_MODES: tuple[ScoringMode, ...] = (
    "entropy_only",
    "lexical_density_only",
    "entity_density_only",
    "concept_density_only",
    "information_gain_only",
    "actionability_only",
    "intrinsic_only",
    "utility_only",
    "full",
)

DETAIL_MODE_KEYS: dict[ScoringMode, str] = {
    "entropy_only": "entropy",
    "lexical_density_only": "lexical_density",
    "entity_density_only": "entity_density",
    "concept_density_only": "concept_density",
    "information_gain_only": "information_gain",
    "actionability_only": "actionability",
}


def normalize_scoring_mode(mode: str | None) -> ScoringMode:
    normalized = (mode or "full").strip().lower()
    if normalized in SCORING_MODES:
        return normalized  # type: ignore[return-value]
    allowed = ", ".join(SCORING_MODES)
    raise ValueError(f"unknown scoring mode: {mode!r}; expected one of: {allowed}")


def select_routing_score(
    mode: ScoringMode,
    *,
    details: dict[str, float],
    intrinsic_score: float,
    utility_score: float,
    full_score: float,
) -> float:
    """Choose the score that should drive threshold routing for a scoring mode."""
    if mode in DETAIL_MODE_KEYS:
        return clamp01(details[DETAIL_MODE_KEYS[mode]])
    if mode == "intrinsic_only":
        return clamp01(intrinsic_score)
    if mode == "utility_only":
        return clamp01(utility_score)
    return clamp01(full_score)

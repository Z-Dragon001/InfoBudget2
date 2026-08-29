"""Strict parsers for JSON responses returned by reference models."""

from __future__ import annotations

import json
import re
from typing import Any

from reference_fact_pipeline.schemas import GroundingDecision, ProposedFact

VALID_FACT_TYPES = {
    "identity", "state", "event", "preference", "goal", "relationship",
    "knowledge", "assistant_answer", "other",
}
VALID_STATE_STATUS = {"current", "historical", "timeless", "unspecified"}


def parse_proposed_facts(
    content: str,
    *,
    segment_turn_ids: set[int],
    origin: str,
    max_facts: int,
) -> list[ProposedFact]:
    payload = _json_object(content)
    key = "missing_facts" if origin == "coverage" else "facts"
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        raise ValueError(f"model response field {key} must be an array")
    facts: list[ProposedFact] = []
    for index, row in enumerate(rows[:max_facts]):
        if not isinstance(row, dict):
            raise ValueError(f"{key}[{index}] must be an object")
        text = " ".join(str(row.get("fact_text") or row.get("text") or "").split())
        if not text:
            continue
        source_ids = tuple(sorted({int(item) for item in row.get("source_turn_ids", ())}))
        if not source_ids or not set(source_ids).issubset(segment_turn_ids):
            continue
        fact_type = str(row.get("fact_type") or "other").strip().lower()
        state_status = str(row.get("state_status") or "unspecified").strip().lower()
        if fact_type not in VALID_FACT_TYPES:
            fact_type = "other"
        if state_status not in VALID_STATE_STATUS:
            state_status = "unspecified"
        facts.append(
            ProposedFact(
                temp_fact_id=f"{origin}_{index:04d}",
                fact_text=text,
                source_turn_ids=source_ids,
                fact_type=fact_type,
                state_status=state_status,
                origin=origin,
                proposal_order=index,
            )
        )
    return facts


def parse_grounding_decisions(
    content: str, proposed_facts: list[ProposedFact]
) -> dict[str, GroundingDecision]:
    payload = _json_object(content)
    rows = payload.get("decisions", [])
    if not isinstance(rows, list):
        raise ValueError("model response field decisions must be an array")
    expected = {item.temp_fact_id for item in proposed_facts}
    decisions: dict[str, GroundingDecision] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"decisions[{index}] must be an object")
        fact_id = str(row.get("temp_fact_id") or "").strip()
        if fact_id not in expected or fact_id in decisions:
            continue
        decision = str(row.get("decision") or "REJECT").strip().upper()
        if decision not in {"ACCEPT", "REJECT"}:
            decision = "REJECT"
        duplicate = row.get("duplicate_of")
        duplicate_of = str(duplicate).strip() if duplicate else None
        if duplicate_of not in expected:
            duplicate_of = None
        decisions[fact_id] = GroundingDecision(
            temp_fact_id=fact_id,
            decision=decision,
            entailed=bool(row.get("entailed", False)),
            atomic=bool(row.get("atomic", False)),
            source_ids_sufficient=bool(row.get("source_ids_sufficient", False)),
            contains_external_inference=bool(row.get("contains_external_inference", True)),
            duplicate_of=duplicate_of,
            reason=str(row.get("reason") or "missing reason").strip(),
        )
    for item in proposed_facts:
        decisions.setdefault(
            item.temp_fact_id,
            GroundingDecision(
                temp_fact_id=item.temp_fact_id,
                decision="REJECT",
                entailed=False,
                atomic=False,
                source_ids_sufficient=False,
                contains_external_inference=True,
                duplicate_of=None,
                reason="grounding judge omitted this proposal",
            ),
        )
    return decisions


def _json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response root must be a JSON object")
    return value


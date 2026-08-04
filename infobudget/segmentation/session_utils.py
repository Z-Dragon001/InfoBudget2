"""Shared helpers for keeping dataset session boundaries during segmentation."""

from __future__ import annotations

from dataclasses import dataclass

from infobudget.schemas import Turn


@dataclass(frozen=True, slots=True)
class SessionSlice:
    """A contiguous run of turns that belongs to one source session."""

    start: int
    end: int
    session_id: str


def contiguous_session_slices(
    turns: list[Turn],
    preserve_boundaries: bool = True,
) -> list[SessionSlice]:
    """Return contiguous session ranges without changing the preprocessed dialogue."""
    if not turns:
        return []
    if not preserve_boundaries:
        return [SessionSlice(0, len(turns), "all")]

    session_ids = [_session_id(turn) for turn in turns]
    slices: list[SessionSlice] = []
    start = 0
    current = session_ids[0]
    for index, session_id in enumerate(session_ids[1:], start=1):
        if session_id == current:
            continue
        slices.append(SessionSlice(start, index, current))
        start = index
        current = session_id
    slices.append(SessionSlice(start, len(turns), current))
    return slices


def _session_id(turn: Turn) -> str:
    value = turn.metadata.get("session_id")
    if value is None or str(value).strip() == "":
        # Consecutive turns without session metadata remain one dialogue, preserving
        # the behavior of ad-hoc and unit-test inputs.
        return "__unscoped__"
    return str(value)

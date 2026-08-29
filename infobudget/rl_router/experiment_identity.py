"""Stable filesystem identities for parameter sweeps."""

from __future__ import annotations


def epoch_artifact_name(epochs: int) -> str:
    value = int(epochs)
    if value <= 0:
        raise ValueError("epochs must be positive")
    return f"epochs_{value}"

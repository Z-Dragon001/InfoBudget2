"""Stable identifiers for parameterized segmentation artifacts."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


METHOD_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def adaptive_alpha_token(value: float | str | Decimal) -> str:
    """Return a filesystem- and namespace-safe canonical alpha token."""

    try:
        alpha = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"adaptive alpha must be a finite number: {value!r}") from exc
    if not alpha.is_finite() or alpha < 0:
        raise ValueError(f"adaptive alpha must be finite and non-negative: {value!r}")
    if alpha == 0:
        alpha = Decimal(0)
    normalized = format(alpha.normalize(), "f")
    return normalized.replace(".", "p")


def segmentation_artifact_name(method: str, adaptive_alpha: float | str | Decimal) -> str:
    """Bind the segmentation algorithm and alpha into one immutable run identity."""

    normalized_method = str(method).strip().lower()
    if not METHOD_PATTERN.fullmatch(normalized_method):
        raise ValueError(f"invalid segmentation method identifier: {method!r}")
    return f"{normalized_method}_alpha_{adaptive_alpha_token(adaptive_alpha)}"


def segmentation_version(method: str, adaptive_alpha: float | str | Decimal) -> str:
    return f"{segmentation_artifact_name(method, adaptive_alpha)}_v1"

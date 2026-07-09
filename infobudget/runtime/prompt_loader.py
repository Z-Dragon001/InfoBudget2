"""Load external prompt templates for the InfoBudget runtime."""

from __future__ import annotations

from pathlib import Path


def load_prompt(prompt_dir: str | Path, name: str) -> str:
    """Read a single prompt template file."""
    path = Path(prompt_dir) / name
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def load_prompt_map(prompt_dir: str | Path, names: dict[str, str]) -> dict[str, str]:
    """Read multiple prompt template files keyed by logical prompt name."""
    return {key: load_prompt(prompt_dir, value) for key, value in names.items()}

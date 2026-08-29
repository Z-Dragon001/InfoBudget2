"""Lightweight terminal progress reporting for long-running experiment stages."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import Any, TextIO


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    value = int(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


class StageProgress:
    """Render a TTY progress bar and readable periodic non-TTY snapshots.

    The class intentionally writes to stderr so stdout can remain suitable for
    machine-readable JSON summaries. It has no dependency on tqdm and is safe
    to use in the repository's subprocess-oriented experiment scripts.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "items",
        initial: int = 0,
        width: int = 24,
        min_interval: float = 0.25,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if total < 0:
            raise ValueError("total must be non-negative")
        if initial < 0 or initial > total:
            raise ValueError("initial must be between zero and total")
        self.label = label
        self.total = total
        self.unit = unit
        self.completed = initial
        self.width = max(8, width)
        self.min_interval = max(0.0, min_interval)
        self.stream = stream or sys.stderr
        self._clock = clock
        self._started_at = clock()
        self._last_rendered_at: float | None = None
        self._last_line_length = 0
        self._closed = False

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def update(
        self,
        advance: int = 1,
        *,
        item: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        if self._closed:
            return
        if advance < 0:
            raise ValueError("advance must be non-negative")
        self.completed = min(self.total, self.completed + advance)
        now = self._clock()
        should_render = (
            force
            or self.completed >= self.total
            or self._last_rendered_at is None
            or now - self._last_rendered_at >= self.min_interval
        )
        if should_render:
            self._render(now, item=item, metrics=metrics)

    def close(
        self,
        *,
        status: str = "done",
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        now = self._clock()
        final_metrics = {"status": status}
        if metrics:
            final_metrics.update(metrics)
        self._render(now, metrics=final_metrics, final=True)
        self._closed = True

    def _render(
        self,
        now: float,
        *,
        item: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        final: bool = False,
    ) -> None:
        elapsed = max(0.0, now - self._started_at)
        ratio = self.completed / self.total if self.total else 1.0
        filled = min(self.width, int(round(ratio * self.width)))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else None
        parts = [
            f"{self.label}: |{bar}|",
            f"{self.completed}/{self.total} {self.unit}",
            f"elapsed={_duration(elapsed)}",
            f"eta={_duration(eta)}",
        ]
        if rate > 0:
            parts.append(f"rate={rate:.2f} {self.unit}/s")
        if item:
            parts.append(f"item={item}")
        if metrics:
            parts.extend(f"{key}={_metric_value(value)}" for key, value in metrics.items())
        line = " ".join(parts)
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if is_tty:
            padding = " " * max(0, self._last_line_length - len(line))
            self.stream.write(f"\r{line}{padding}")
            if final:
                self.stream.write("\n")
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()
        self._last_line_length = len(line)
        self._last_rendered_at = now

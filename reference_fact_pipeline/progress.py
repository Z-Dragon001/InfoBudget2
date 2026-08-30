"""Realtime terminal progress for frozen-reference construction."""

from __future__ import annotations

from typing import Any, Mapping, TextIO

from infobudget.utils.progress import StageProgress


class ReferenceFactProgress:
    """Keep stdout JSON-clean while reporting segment, stage, wait, and token state."""

    def __init__(self, total_segments: int, *, stream: TextIO | None = None) -> None:
        self._bar = StageProgress(
            "Gold Fact", total_segments, unit="segments", stream=stream
        )
        self.current_segment = ""
        self.current_stage = "starting"
        self.built = 0
        self.skipped = 0
        self.failed = 0
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def start_segment(self, segment_id: str) -> None:
        self.current_segment = segment_id
        self.current_stage = "starting"
        self._render(force=True)

    def handle_event(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("event") or "")
        if kind == "stage_started":
            self.current_stage = str(event.get("stage") or "unknown")
            self._render(force=True)
        elif kind == "stage_completed":
            self.current_stage = str(event.get("stage") or "unknown")
            self.api_calls += 1
            self.input_tokens += int(event.get("input_tokens") or 0)
            self.output_tokens += int(event.get("output_tokens") or 0)
            self._render(force=True)
        elif kind == "stage_failed":
            self.current_stage = f"{event.get('stage') or 'unknown'}:failed"
            self._render(force=True)
        elif kind == "wait_started":
            reason = str(event.get("reason") or "wait")
            seconds = float(event.get("seconds") or 0.0)
            self.current_stage = f"{reason}:{seconds:.0f}s"
            self._render(force=True)
        elif kind == "wait_finished":
            self.current_stage = str(event.get("resume_stage") or "resuming")
            self._render(force=True)

    def segment_finished(
        self, status: str, *, existing_row: Mapping[str, Any] | None = None
    ) -> None:
        if status == "built":
            self.built += 1
        elif status == "skipped":
            self.skipped += 1
            if existing_row is not None:
                self.input_tokens += int(existing_row.get("total_input_tokens") or 0)
                self.output_tokens += int(existing_row.get("total_output_tokens") or 0)
                self.api_calls += len(existing_row.get("stage_usage") or ())
        elif status == "failed":
            self.failed += 1
        self.current_stage = status
        self._bar.update(
            1,
            item=self.current_segment,
            metrics=self._metrics(),
            force=True,
        )

    def close(self, *, paused: bool, incomplete: bool = False) -> None:
        status = "paused" if paused else "incomplete" if incomplete else "done"
        self._bar.close(
            status=status,
            metrics=self._metrics(),
        )

    def _render(self, *, force: bool) -> None:
        self._bar.update(
            0,
            item=self.current_segment or None,
            metrics=self._metrics(),
            force=force,
        )

    def _metrics(self) -> dict[str, Any]:
        return {
            "stage": self.current_stage,
            "built": self.built,
            "skipped": self.skipped,
            "failed": self.failed,
            "calls": self.api_calls,
            "input_tok": self.input_tokens,
            "output_tok": self.output_tokens,
        }

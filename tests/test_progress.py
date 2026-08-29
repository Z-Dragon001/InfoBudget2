from __future__ import annotations

import io

import pytest

from infobudget.utils.progress import StageProgress


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_stage_progress_reports_counts_eta_and_metrics() -> None:
    output = io.StringIO()
    clock = FakeClock()
    progress = StageProgress(
        "candidate extraction",
        2,
        unit="batches",
        stream=output,
        clock=clock,
        min_interval=0,
    )

    clock.value = 2.0
    progress.update(item="small:committed", metrics={"tokens": 120, "cost": 0.0012})
    clock.value = 4.0
    progress.update(item="large:committed", metrics={"tokens": 240, "cost": 0.0024})
    progress.close(metrics={"facts": 8})

    rendered = output.getvalue()
    assert "candidate extraction:" in rendered
    assert "1/2 batches" in rendered
    assert "2/2 batches" in rendered
    assert "eta=00:02" in rendered
    assert "item=small:committed" in rendered
    assert "tokens=120" in rendered
    assert "facts=8" in rendered
    assert "status=done" in rendered


def test_stage_progress_validates_bounds() -> None:
    with pytest.raises(ValueError, match="total"):
        StageProgress("invalid", -1)
    with pytest.raises(ValueError, match="initial"):
        StageProgress("invalid", 2, initial=3)


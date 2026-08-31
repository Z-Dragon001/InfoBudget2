"""Local epsilon-noninferiority routing over predicted Fact quality."""

from infobudget.quality_gap_router.calibration import (
    CalibrationResult,
    calibrate_quality_gap,
    estimate_gap_residual_bound,
)
from infobudget.quality_gap_router.decision import (
    QualityGapDecision,
    QualityGapPolicy,
    select_quality_gap_model,
)
from infobudget.quality_gap_router.evaluation import (
    QualityGapEvaluation,
    evaluate_quality_gap,
)
from infobudget.quality_gap_router.schemas import QualityObservation

__all__ = [
    "CalibrationResult",
    "QualityGapDecision",
    "QualityGapEvaluation",
    "QualityGapPolicy",
    "QualityObservation",
    "calibrate_quality_gap",
    "estimate_gap_residual_bound",
    "evaluate_quality_gap",
    "select_quality_gap_model",
]

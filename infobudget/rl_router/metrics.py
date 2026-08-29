"""Paper-facing aggregate metrics for routed experiments."""

from __future__ import annotations

import math
import statistics


def summarize_fold_accuracy(values) -> dict:
    accuracies = [float(value) for value in values]
    if not accuracies:
        return {
            "fold_qa_accuracy_micro": [],
            "mean_fold_accuracy_micro": 0.0,
            "std_fold_accuracy_micro": 0.0,
            "sem_fold_accuracy_micro": 0.0,
            "min_fold_accuracy_micro": 0.0,
            "max_fold_accuracy_micro": 0.0,
        }
    deviation = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    return {
        "fold_qa_accuracy_micro": accuracies,
        "mean_fold_accuracy_micro": statistics.fmean(accuracies),
        "std_fold_accuracy_micro": deviation,
        "sem_fold_accuracy_micro": deviation / math.sqrt(len(accuracies)),
        "min_fold_accuracy_micro": min(accuracies),
        "max_fold_accuracy_micro": max(accuracies),
    }

"""Reusable experiment harnesses for comparing training methods and architectures."""

from fabricpc.experiments.ab_experiment import ExperimentArm, ABExperiment, ABResults
from fabricpc.experiments.statistics import (
    paired_ttest,
    cohens_d,
    estimate_required_n,
    descriptive_stats,
)

__all__ = [
    "ExperimentArm",
    "ABExperiment",
    "ABResults",
    "paired_ttest",
    "cohens_d",
    "estimate_required_n",
    "descriptive_stats",
]

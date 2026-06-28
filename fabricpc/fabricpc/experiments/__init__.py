"""Reusable experiment harnesses for comparing training methods and architectures."""

from fabricpc.experiments.ab_experiment import ABExperiment, ABResults, ExperimentArm
from fabricpc.experiments.statistics import (
    cohens_d,
    descriptive_stats,
    estimate_required_n,
    paired_ttest,
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

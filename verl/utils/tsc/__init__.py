"""
Tool-Grounded Self-Consistency (TSC) utilities for VERL.

This package provides tools for implementing self-consistency-based rewards
grounded in external verification tools (unit tests, calculators, constraint checkers).
"""

from .clustering import cluster_candidates, Cluster
from .tool_evaluation import (
    ToolEvaluationResult,
    ToolEvaluator,
    CodeToolEvaluator,
    MathToolEvaluator,
    evaluate_with_tools,
)
from .reward_computation import compute_tsc_rewards, TSCConfig

__all__ = [
    'cluster_candidates',
    'Cluster',
    'ToolEvaluationResult',
    'ToolEvaluator',
    'CodeToolEvaluator',
    'MathToolEvaluator',
    'evaluate_with_tools',
    'compute_tsc_rewards',
    'TSCConfig',
]

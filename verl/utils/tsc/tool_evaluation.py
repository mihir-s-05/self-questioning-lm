"""
Tool-based evaluation interface for self-consistency rewards.

This module provides a generic interface for evaluating candidate solutions
using external tools (unit tests, calculators, constraint checkers, etc.).
It wraps existing reward scoring functions from verl.utils.reward_score.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import traceback

logger = logging.getLogger(__name__)


@dataclass
class ToolEvaluationResult:
    """Result of tool-based evaluation for a single candidate.

    Attributes:
        score: Continuous score in [0, 1], where 1.0 means fully correct
        passed: Boolean indicating whether the solution is considered correct
        details: Optional dictionary with additional diagnostic information
        error: Optional error message if evaluation failed
        test_results: Optional list of per-test results (for code evaluation)
    """
    score: float
    passed: bool
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    test_results: Optional[List[bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with clustering."""
        result = {
            'score': self.score,
            'passed': self.passed,
        }
        if self.test_results is not None:
            result['test_results'] = self.test_results
        if self.details is not None:
            result['details'] = self.details
        if self.error is not None:
            result['error'] = self.error
        return result


class ToolEvaluator(ABC):
    """Abstract base class for tool-based evaluators."""

    @abstractmethod
    def evaluate(
        self,
        problem: str,
        candidate: str,
        ground_truth: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ToolEvaluationResult:
        """Evaluate a candidate solution using external tools.

        Args:
            problem: The problem statement or query
            candidate: The candidate solution to evaluate
            ground_truth: Optional ground truth answer for comparison
            extra_info: Optional additional information needed for evaluation

        Returns:
            ToolEvaluationResult with score and diagnostics
        """
        pass


class MathToolEvaluator(ToolEvaluator):
    """Evaluator for math problems using existing VERL math scorers.

    Supports:
    - MATH dataset (with LaTeX parsing)
    - GSM8K (arithmetic word problems)
    - Other numeric answer formats
    """

    def __init__(self, data_source: str = "lighteval/MATH"):
        """Initialize math evaluator.

        Args:
            data_source: Dataset identifier (e.g., "lighteval/MATH", "openai/gsm8k")
        """
        self.data_source = data_source

    def evaluate(
        self,
        problem: str,
        candidate: str,
        ground_truth: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ToolEvaluationResult:
        """Evaluate math solution against ground truth.

        Args:
            problem: Math problem statement (not used directly)
            candidate: Candidate solution string
            ground_truth: Ground truth answer
            extra_info: Additional information

        Returns:
            ToolEvaluationResult with binary score (0.0 or 1.0)
        """
        if ground_truth is None:
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error="Ground truth required for math evaluation",
            )

        try:
            from verl.utils.reward_score import default_compute_score

            score = default_compute_score(
                data_source=self.data_source,
                solution_str=candidate,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # Math scoring is typically binary (0.0 or 1.0)
            passed = score >= 0.5

            return ToolEvaluationResult(
                score=float(score),
                passed=passed,
                details={'data_source': self.data_source},
            )

        except Exception as e:
            logger.error(f"Math evaluation failed: {e}")
            traceback.print_exc()
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error=str(e),
            )


class CodeToolEvaluator(ToolEvaluator):
    """Evaluator for code problems using sandbox execution.

    Runs candidate code against test cases in a sandboxed environment.
    Supports partial credit based on the fraction of tests passed.
    """

    def __init__(
        self,
        data_source: str = "codecontests",
        sandbox_fusion_url: Optional[str] = None,
        concurrent_semaphore: Optional[Any] = None,
        timeout: float = 10.0,
        continuous: bool = True,
    ):
        """Initialize code evaluator.

        Args:
            data_source: Dataset identifier (e.g., "codecontests", "apps")
            sandbox_fusion_url: URL of sandbox service (if None, uses local execution)
            concurrent_semaphore: Optional semaphore for concurrent execution control
            timeout: Timeout per test case in seconds
            continuous: Whether to compute continuous score (partial credit)
        """
        self.data_source = data_source
        self.sandbox_fusion_url = sandbox_fusion_url
        self.concurrent_semaphore = concurrent_semaphore
        self.timeout = timeout
        self.continuous = continuous

    def evaluate(
        self,
        problem: str,
        candidate: str,
        ground_truth: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ToolEvaluationResult:
        """Evaluate code solution by running test cases.

        Args:
            problem: Problem statement (may contain test cases)
            candidate: Candidate code solution
            ground_truth: Test cases (JSON string or dict with 'inputs'/'outputs')
            extra_info: Additional information

        Returns:
            ToolEvaluationResult with continuous score based on tests passed
        """
        if ground_truth is None:
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error="Test cases (ground_truth) required for code evaluation",
            )

        try:
            from verl.utils.reward_score import default_compute_score

            # Call the sandbox evaluation
            result = default_compute_score(
                data_source=self.data_source,
                solution_str=candidate,
                ground_truth=ground_truth,
                extra_info=extra_info,
                sandbox_fusion_url=self.sandbox_fusion_url,
                concurrent_semaphore=self.concurrent_semaphore,
            )

            # Result can be a float (score) or a tuple (score, metadata)
            if isinstance(result, tuple):
                score, metadata = result
                # Extract per-test results if available
                test_results = None
                if isinstance(metadata, list):
                    # metadata_list from sandbox contains execution info
                    test_results = [
                        m.get('passed', False) if isinstance(m, dict) else False
                        for m in metadata
                    ]
                passed = score >= 0.8  # Consider passing if 80%+ tests pass
                return ToolEvaluationResult(
                    score=float(score),
                    passed=passed,
                    details={'metadata': metadata},
                    test_results=test_results,
                )
            else:
                # Simple float score
                score = float(result)
                passed = score >= 0.8
                return ToolEvaluationResult(
                    score=score,
                    passed=passed,
                )

        except Exception as e:
            logger.error(f"Code evaluation failed: {e}")
            traceback.print_exc()
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error=str(e),
            )


class QAToolEvaluator(ToolEvaluator):
    """Evaluator for QA tasks using exact match or F1 scoring.

    Supports various QA datasets with string matching and normalization.
    """

    def __init__(self, data_source: str):
        """Initialize QA evaluator.

        Args:
            data_source: Dataset identifier (e.g., "searchR1_nq", "searchR1_triviaqa")
        """
        self.data_source = data_source

    def evaluate(
        self,
        problem: str,
        candidate: str,
        ground_truth: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ToolEvaluationResult:
        """Evaluate QA solution using exact match.

        Args:
            problem: Question
            candidate: Candidate answer
            ground_truth: Ground truth answer
            extra_info: Additional information

        Returns:
            ToolEvaluationResult with binary score
        """
        if ground_truth is None:
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error="Ground truth required for QA evaluation",
            )

        try:
            from verl.utils.reward_score import default_compute_score

            score = default_compute_score(
                data_source=self.data_source,
                solution_str=candidate,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            passed = score >= 0.5

            return ToolEvaluationResult(
                score=float(score),
                passed=passed,
                details={'data_source': self.data_source},
            )

        except Exception as e:
            logger.error(f"QA evaluation failed: {e}")
            traceback.print_exc()
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error=str(e),
            )


class CustomToolEvaluator(ToolEvaluator):
    """Evaluator using a custom scoring function.

    Allows users to provide their own evaluation logic.
    """

    def __init__(self, scoring_fn):
        """Initialize with custom scoring function.

        Args:
            scoring_fn: Callable that takes (problem, candidate, ground_truth, extra_info)
                       and returns a float score in [0, 1]
        """
        self.scoring_fn = scoring_fn

    def evaluate(
        self,
        problem: str,
        candidate: str,
        ground_truth: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ToolEvaluationResult:
        """Evaluate using custom scoring function.

        Args:
            problem: Problem statement
            candidate: Candidate solution
            ground_truth: Ground truth (optional)
            extra_info: Additional information (optional)

        Returns:
            ToolEvaluationResult
        """
        try:
            score = self.scoring_fn(problem, candidate, ground_truth, extra_info)
            score = float(score)
            passed = score >= 0.5

            return ToolEvaluationResult(
                score=score,
                passed=passed,
                details={'custom': True},
            )

        except Exception as e:
            logger.error(f"Custom evaluation failed: {e}")
            traceback.print_exc()
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error=str(e),
            )


def create_evaluator(
    tool_type: str,
    data_source: Optional[str] = None,
    **kwargs
) -> ToolEvaluator:
    """Factory function to create appropriate evaluator.

    Args:
        tool_type: Type of tool ("math", "code", "qa", "custom")
        data_source: Dataset identifier (required for most types)
        **kwargs: Additional arguments passed to evaluator constructor

    Returns:
        ToolEvaluator instance

    Raises:
        ValueError: If tool_type is unknown
    """
    if tool_type == "math":
        data_source = data_source or "lighteval/MATH"
        return MathToolEvaluator(data_source=data_source)

    elif tool_type == "code":
        data_source = data_source or "codecontests"
        return CodeToolEvaluator(data_source=data_source, **kwargs)

    elif tool_type == "qa":
        if data_source is None:
            raise ValueError("data_source required for QA evaluator")
        return QAToolEvaluator(data_source=data_source)

    elif tool_type == "custom":
        if 'scoring_fn' not in kwargs:
            raise ValueError("scoring_fn required for custom evaluator")
        return CustomToolEvaluator(**kwargs)

    else:
        raise ValueError(f"Unknown tool_type: {tool_type}")


def evaluate_with_tools(
    evaluator: ToolEvaluator,
    problem: str,
    candidate: str,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    safe_fallback: bool = True,
) -> ToolEvaluationResult:
    """Convenience function to evaluate a candidate with an evaluator.

    Provides safe fallback behavior on errors to prevent training crashes.

    Args:
        evaluator: ToolEvaluator instance
        problem: Problem statement
        candidate: Candidate solution
        ground_truth: Ground truth answer
        extra_info: Additional information
        safe_fallback: If True, return zero score on error instead of raising

    Returns:
        ToolEvaluationResult
    """
    try:
        return evaluator.evaluate(problem, candidate, ground_truth, extra_info)
    except Exception as e:
        logger.error(f"Tool evaluation error: {e}")
        if safe_fallback:
            logger.warning("Using safe fallback: score=0.0")
            return ToolEvaluationResult(
                score=0.0,
                passed=False,
                error=f"Evaluation failed with safe fallback: {str(e)}",
            )
        else:
            raise

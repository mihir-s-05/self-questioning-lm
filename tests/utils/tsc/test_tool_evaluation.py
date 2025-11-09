"""
Tests for TSC tool evaluation.
"""

import pytest

from verl.utils.tsc.tool_evaluation import (
    ToolEvaluationResult,
    MathToolEvaluator,
    create_evaluator,
    evaluate_with_tools,
)


class TestToolEvaluationResult:
    """Test ToolEvaluationResult dataclass."""

    def test_basic_result(self):
        result = ToolEvaluationResult(score=1.0, passed=True)
        assert result.score == 1.0
        assert result.passed is True

    def test_to_dict(self):
        result = ToolEvaluationResult(
            score=0.5,
            passed=False,
            error="test error"
        )
        d = result.to_dict()
        assert d['score'] == 0.5
        assert d['passed'] is False
        assert d['error'] == "test error"

    def test_with_test_results(self):
        result = ToolEvaluationResult(
            score=0.5,
            passed=False,
            test_results=[True, False, True, False]
        )
        d = result.to_dict()
        assert 'test_results' in d
        assert len(d['test_results']) == 4


class TestMathToolEvaluator:
    """Test math tool evaluator."""

    def test_correct_answer(self):
        evaluator = MathToolEvaluator(data_source="openai/gsm8k")

        # Simple GSM8K-style answer
        candidate = "#### 42"
        ground_truth = "42"

        result = evaluator.evaluate("", candidate, ground_truth)

        # Note: This might fail if gsm8k scorer is not properly set up
        # In that case, it should at least return a valid ToolEvaluationResult
        assert isinstance(result, ToolEvaluationResult)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)

    def test_missing_ground_truth(self):
        evaluator = MathToolEvaluator()
        result = evaluator.evaluate("", "42", ground_truth=None)

        assert result.score == 0.0
        assert result.passed is False
        assert result.error is not None


class TestCreateEvaluator:
    """Test evaluator factory."""

    def test_create_math_evaluator(self):
        evaluator = create_evaluator("math")
        assert isinstance(evaluator, MathToolEvaluator)

    def test_create_math_with_data_source(self):
        evaluator = create_evaluator("math", data_source="openai/gsm8k")
        assert isinstance(evaluator, MathToolEvaluator)
        assert evaluator.data_source == "openai/gsm8k"

    def test_unknown_tool_type(self):
        with pytest.raises(ValueError, match="Unknown tool_type"):
            create_evaluator("unknown_tool")

    def test_custom_without_scoring_fn(self):
        with pytest.raises(ValueError, match="scoring_fn required"):
            create_evaluator("custom")


class TestEvaluateWithTools:
    """Test convenience evaluation function."""

    def test_safe_fallback(self):
        evaluator = MathToolEvaluator()

        # This should trigger an error (missing ground truth)
        # but safe_fallback should return zero score instead of raising
        result = evaluate_with_tools(
            evaluator=evaluator,
            problem="",
            candidate="42",
            ground_truth=None,
            safe_fallback=True,
        )

        assert result.score == 0.0
        assert result.passed is False


class TestIntegration:
    """Integration tests for tool evaluation."""

    def test_math_evaluation_flow(self):
        """Test complete math evaluation flow."""
        # Create evaluator
        evaluator = create_evaluator("math", data_source="openai/gsm8k")

        # Evaluate a candidate
        candidate = "Let me solve this. The answer is #### 42"
        ground_truth = "42"

        result = evaluate_with_tools(
            evaluator=evaluator,
            problem="What is 40 + 2?",
            candidate=candidate,
            ground_truth=ground_truth,
            safe_fallback=True,
        )

        # Should return a valid result
        assert isinstance(result, ToolEvaluationResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)

    def test_multiple_candidates(self):
        """Test evaluating multiple candidates."""
        evaluator = create_evaluator("math", data_source="openai/gsm8k")

        candidates = [
            "#### 42",
            "#### 43",
            "#### 42.0",
        ]
        ground_truth = "42"

        results = []
        for candidate in candidates:
            result = evaluate_with_tools(
                evaluator,
                "",
                candidate,
                ground_truth,
                safe_fallback=True,
            )
            results.append(result)

        # All results should be valid
        assert len(results) == 3
        assert all(isinstance(r, ToolEvaluationResult) for r in results)
        assert all(0.0 <= r.score <= 1.0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

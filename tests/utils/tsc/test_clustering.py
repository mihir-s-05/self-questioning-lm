"""
Tests for TSC clustering module.
"""

import pytest
import numpy as np

from verl.utils.tsc.clustering import (
    cluster_candidates,
    normalize_text,
    extract_numeric_answer,
    create_tool_signature,
    cluster_by_tool_behavior,
    cluster_by_normalized_text,
    cluster_by_numeric_answer,
)


class TestNormalizeText:
    """Test text normalization."""

    def test_basic_normalization(self):
        text = "  Hello   World  "
        assert normalize_text(text) == "hello world"

    def test_math_normalization(self):
        text = "The answer is $42.00"
        normalized = normalize_text(text)
        assert "42" in normalized
        assert "$" not in normalized

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestExtractNumericAnswer:
    """Test numeric answer extraction."""

    def test_plain_number(self):
        assert extract_numeric_answer("42") == 42.0
        assert extract_numeric_answer("3.14") == 3.14

    def test_boxed_answer(self):
        assert extract_numeric_answer("\\boxed{42}") == 42.0

    def test_answer_statement(self):
        assert extract_numeric_answer("The answer is 42") == 42.0

    def test_fraction(self):
        result = extract_numeric_answer("1/2")
        assert result is not None
        assert abs(result - 0.5) < 1e-6

    def test_no_number(self):
        assert extract_numeric_answer("no numbers here") is None


class TestCreateToolSignature:
    """Test tool signature creation."""

    def test_simple_signature(self):
        tool_result = {'score': 1.0, 'passed': True}
        sig = create_tool_signature(tool_result)
        assert sig == (True, 1.0)

    def test_signature_with_tests(self):
        tool_result = {
            'score': 0.5,
            'passed': False,
            'test_results': [True, False, True, False]
        }
        sig = create_tool_signature(tool_result)
        assert sig == (False, 0.5, (True, False, True, False))


class TestClusterByToolBehavior:
    """Test clustering by tool behavior."""

    def test_single_cluster_all_correct(self):
        candidates = ["solution 1", "solution 2", "solution 3"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
        ]

        clusters = cluster_by_tool_behavior(candidates, tool_results)

        assert len(clusters) == 1
        assert clusters[0].size == 3
        assert clusters[0].is_tool_correct is True

    def test_multiple_clusters(self):
        candidates = ["sol1", "sol2", "sol3", "sol4"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
            {'score': 0.5, 'passed': False},
        ]

        clusters = cluster_by_tool_behavior(candidates, tool_results)

        # Should have 3 clusters (1.0, 0.0, 0.5)
        assert len(clusters) == 3


class TestClusterByNormalizedText:
    """Test clustering by normalized text."""

    def test_identical_after_normalization(self):
        candidates = ["  Answer: 42  ", "Answer: 42", "answer:42"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
        ]

        clusters = cluster_by_normalized_text(candidates, tool_results)

        # All should be in same cluster after normalization
        assert len(clusters) == 1
        assert clusters[0].size == 3

    def test_different_answers(self):
        candidates = ["42", "43", "44"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
            {'score': 0.0, 'passed': False},
        ]

        clusters = cluster_by_normalized_text(candidates, tool_results)

        # Should have 3 clusters
        assert len(clusters) == 3


class TestClusterByNumericAnswer:
    """Test clustering by numeric answer."""

    def test_same_numeric_value(self):
        candidates = ["42", "42.0", "The answer is 42"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
        ]

        clusters = cluster_by_numeric_answer(candidates, tool_results)

        # All should be in same cluster
        assert len(clusters) == 1
        assert clusters[0].size == 3

    def test_different_numeric_values(self):
        candidates = ["42", "43", "44"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
            {'score': 0.0, 'passed': False},
        ]

        clusters = cluster_by_numeric_answer(candidates, tool_results)

        # Should have 3 clusters
        assert len(clusters) == 3

    def test_tolerance(self):
        candidates = ["1.0000001", "1.0000002"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
        ]

        clusters = cluster_by_numeric_answer(candidates, tool_results, tolerance=1e-5)

        # Should be in same cluster due to tolerance
        assert len(clusters) == 1


class TestClusterCandidates:
    """Test main clustering function."""

    def test_auto_mode_with_tests(self):
        candidates = ["code1", "code2", "code3"]
        tool_results = [
            {'score': 1.0, 'passed': True, 'test_results': [True, True]},
            {'score': 1.0, 'passed': True, 'test_results': [True, True]},
            {'score': 0.5, 'passed': False, 'test_results': [True, False]},
        ]

        clusters = cluster_candidates(candidates, tool_results, mode='auto')

        # Auto should detect tests and use tool-based clustering
        # Should have 2 clusters (different test results)
        assert len(clusters) == 2

    def test_auto_mode_with_math(self):
        candidates = ["42", "42.0", "43"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
        ]

        clusters = cluster_candidates(candidates, tool_results, mode='auto')

        # Auto should detect numeric and use numeric clustering
        # Should have 2 clusters (42 and 43)
        assert len(clusters) == 2

    def test_empty_input(self):
        clusters = cluster_candidates([], [], mode='auto')
        assert len(clusters) == 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="must have the same length"):
            cluster_candidates(["a", "b"], [{'score': 1.0}], mode='auto')

    def test_sorted_by_size(self):
        candidates = ["a", "b", "c", "d", "e"]
        tool_results = [
            {'score': 1.0, 'passed': True},  # cluster 1
            {'score': 1.0, 'passed': True},  # cluster 1
            {'score': 1.0, 'passed': True},  # cluster 1
            {'score': 0.0, 'passed': False},  # cluster 2
            {'score': 0.5, 'passed': False},  # cluster 3
        ]

        clusters = cluster_by_tool_behavior(candidates, tool_results)

        # Clusters should be sorted by size (descending)
        assert clusters[0].size >= clusters[1].size
        assert clusters[1].size >= clusters[2].size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for TSC reward computation.
"""

import pytest
import numpy as np

from verl.utils.tsc.reward_computation import (
    TSCConfig,
    compute_tsc_rewards,
    compute_tsc_metrics,
    identify_best_cluster,
    is_majority_cluster,
    is_cluster_correct,
)
from verl.utils.tsc.clustering import Cluster


class TestTSCConfig:
    """Test TSC configuration."""

    def test_default_config(self):
        config = TSCConfig()
        assert config.alpha_tool == 0.8
        assert config.beta_consistency == 0.2
        assert config.normalize_rewards is True

    def test_custom_config(self):
        config = TSCConfig(
            alpha_tool=0.9,
            beta_consistency=0.1,
            correct_majority_bonus=0.2,
        )
        assert config.alpha_tool == 0.9
        assert config.beta_consistency == 0.1
        assert config.correct_majority_bonus == 0.2

    def test_validate_valid_config(self):
        config = TSCConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_alpha(self):
        config = TSCConfig(alpha_tool=1.5)
        with pytest.raises(ValueError, match="alpha_tool"):
            config.validate()

    def test_validate_invalid_threshold(self):
        config = TSCConfig(min_cluster_frac_for_majority=1.5)
        with pytest.raises(ValueError, match="min_cluster_frac_for_majority"):
            config.validate()


class TestIdentifyBestCluster:
    """Test best cluster identification."""

    def test_single_cluster(self):
        cluster = Cluster(
            members=[0, 1],
            size=2,
            avg_tool_score=0.8,
            is_tool_correct=True,
            representative_idx=0,
        )
        best = identify_best_cluster([cluster])
        assert best == cluster

    def test_best_by_score(self):
        c1 = Cluster([0], 1, 0.5, False, 0)
        c2 = Cluster([1, 2], 2, 0.9, True, 1)
        c3 = Cluster([3], 1, 0.7, False, 3)

        best = identify_best_cluster([c1, c2, c3])
        assert best == c2  # Highest avg_tool_score

    def test_tiebreak_by_size(self):
        c1 = Cluster([0], 1, 1.0, True, 0)
        c2 = Cluster([1, 2], 2, 1.0, True, 1)

        best = identify_best_cluster([c1, c2])
        assert best == c2  # Same score, larger size

    def test_empty_list(self):
        with pytest.raises(ValueError, match="empty list"):
            identify_best_cluster([])


class TestIsMajorityCluster:
    """Test majority cluster detection."""

    def test_is_majority(self):
        cluster = Cluster([0, 1, 2], 3, 1.0, True, 0)
        assert is_majority_cluster(cluster, 4, 0.5) is True

    def test_not_majority(self):
        cluster = Cluster([0, 1], 2, 1.0, True, 0)
        assert is_majority_cluster(cluster, 5, 0.5) is False

    def test_exact_threshold(self):
        cluster = Cluster([0, 1], 2, 1.0, True, 0)
        assert is_majority_cluster(cluster, 4, 0.5) is True


class TestIsClusterCorrect:
    """Test cluster correctness detection."""

    def test_is_correct(self):
        cluster = Cluster([0, 1], 2, 0.9, True, 0)
        assert is_cluster_correct(cluster, 0.8) is True

    def test_not_correct(self):
        cluster = Cluster([0, 1], 2, 0.7, False, 0)
        assert is_cluster_correct(cluster, 0.8) is False

    def test_exact_threshold(self):
        cluster = Cluster([0, 1], 2, 0.8, True, 0)
        assert is_cluster_correct(cluster, 0.8) is True


class TestComputeTSCRewards:
    """Test TSC reward computation."""

    def test_correct_majority(self):
        """Test reward for correct majority."""
        candidates = ["4", "4", "4", "5"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
        ]

        # Create clusters
        c1 = Cluster([0, 1, 2], 3, 1.0, True, 0)
        c2 = Cluster([3], 1, 0.0, False, 3)
        clusters = [c1, c2]

        config = TSCConfig(normalize_rewards=False)
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

        # Correct majority should get high reward
        assert all(rewards[i] > rewards[3] for i in range(3))

    def test_wrong_majority(self):
        """Test penalty for wrong majority."""
        candidates = ["5", "5", "5", "4"]
        tool_results = [
            {'score': 0.0, 'passed': False},
            {'score': 0.0, 'passed': False},
            {'score': 0.0, 'passed': False},
            {'score': 1.0, 'passed': True},
        ]

        # Create clusters
        c1 = Cluster([0, 1, 2], 3, 0.0, False, 0)
        c2 = Cluster([3], 1, 1.0, True, 3)
        clusters = [c1, c2]

        config = TSCConfig(normalize_rewards=False)
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

        # Wrong majority should get penalty
        assert all(rewards[i] < rewards[3] for i in range(3))

    def test_correct_minority_reward(self):
        """Test correct minority when majority is wrong."""
        candidates = ["5", "5", "5", "4"]
        tool_results = [
            {'score': 0.0, 'passed': False},
            {'score': 0.0, 'passed': False},
            {'score': 0.0, 'passed': False},
            {'score': 1.0, 'passed': True},
        ]

        c1 = Cluster([0, 1, 2], 3, 0.0, False, 0)
        c2 = Cluster([3], 1, 1.0, True, 3)
        clusters = [c1, c2]

        config = TSCConfig(
            normalize_rewards=False,
            allow_correct_minority_reward=True,
            correct_minority_bonus=0.5,
        )
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

        # Correct minority should get bonus
        assert rewards[3] > config.alpha_tool * 1.0

    def test_normalize_rewards(self):
        """Test reward normalization."""
        candidates = ["4", "4", "5"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
        ]

        c1 = Cluster([0, 1], 2, 1.0, True, 0)
        c2 = Cluster([2], 1, 0.0, False, 2)
        clusters = [c1, c2]

        config = TSCConfig(normalize_rewards=True)
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

        # Mean should be close to 0
        assert abs(np.mean(rewards)) < 1e-5

    def test_reward_scaling(self):
        """Test reward scaling."""
        candidates = ["4", "5"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
        ]

        c1 = Cluster([0], 1, 1.0, True, 0)
        c2 = Cluster([1], 1, 0.0, False, 1)
        clusters = [c1, c2]

        config1 = TSCConfig(normalize_rewards=False, reward_scale=1.0)
        config2 = TSCConfig(normalize_rewards=False, reward_scale=2.0)

        rewards1 = compute_tsc_rewards(candidates, tool_results, clusters, config1)
        rewards2 = compute_tsc_rewards(candidates, tool_results, clusters, config2)

        # Rewards should be scaled
        assert abs(rewards2[0] - 2 * rewards1[0]) < 1e-5

    def test_length_mismatch(self):
        """Test error on length mismatch."""
        with pytest.raises(ValueError, match="must have same length"):
            compute_tsc_rewards(
                ["a", "b"],
                [{'score': 1.0}],
                [],
                TSCConfig()
            )

    def test_empty_input(self):
        """Test empty input."""
        rewards = compute_tsc_rewards([], [], [], TSCConfig())
        assert len(rewards) == 0


class TestComputeTSCMetrics:
    """Test TSC metrics computation."""

    def test_basic_metrics(self):
        """Test basic metric computation."""
        candidates = ["4", "4", "5"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
        ]

        c1 = Cluster([0, 1], 2, 1.0, True, 0)
        c2 = Cluster([2], 1, 0.0, False, 2)
        clusters = [c1, c2]

        config = TSCConfig()
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)
        metrics = compute_tsc_metrics(candidates, tool_results, clusters, rewards, config)

        # Check basic metrics exist
        assert 'avg_tool_score' in metrics
        assert 'num_clusters' in metrics
        assert 'cluster_entropy' in metrics

        # Check values
        assert metrics['num_clusters'] == 2
        assert abs(metrics['avg_tool_score'] - 2.0/3.0) < 1e-5

    def test_majority_metrics(self):
        """Test majority-specific metrics."""
        candidates = ["4", "4", "4", "5"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
        ]

        c1 = Cluster([0, 1, 2], 3, 1.0, True, 0)
        c2 = Cluster([3], 1, 0.0, False, 3)
        clusters = [c1, c2]

        config = TSCConfig()
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)
        metrics = compute_tsc_metrics(candidates, tool_results, clusters, rewards, config)

        assert metrics['has_majority'] == 1.0
        assert metrics['majority_is_correct'] == 1.0
        assert metrics['frac_correct_majority'] == 0.75
        assert metrics['frac_wrong_majority'] == 0.0

    def test_empty_input(self):
        """Test empty input."""
        metrics = compute_tsc_metrics([], [], [], [], TSCConfig())
        assert metrics == {}


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_math_problem_scenario(self):
        """Test a realistic math problem scenario."""
        # Problem: 2+2, K=4 candidates
        candidates = ["4", "4 ", "5", "3"]
        tool_results = [
            {'score': 1.0, 'passed': True},
            {'score': 1.0, 'passed': True},
            {'score': 0.0, 'passed': False},
            {'score': 0.0, 'passed': False},
        ]

        # Cluster by normalized text
        from verl.utils.tsc.clustering import cluster_by_normalized_text
        clusters = cluster_by_normalized_text(candidates, tool_results)

        config = TSCConfig(normalize_rewards=False)
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

        # "4" and "4 " should be in same cluster and get high rewards
        assert abs(rewards[0] - rewards[1]) < 1e-5
        assert rewards[0] > rewards[2]
        assert rewards[0] > rewards[3]

    def test_code_problem_scenario(self):
        """Test a realistic code problem scenario."""
        # Problem: implement function, K=3 candidates
        candidates = ["code1", "code2", "code3"]
        tool_results = [
            {'score': 1.0, 'passed': True, 'test_results': [True, True, True]},
            {'score': 1.0, 'passed': True, 'test_results': [True, True, True]},
            {'score': 0.33, 'passed': False, 'test_results': [True, False, False]},
        ]

        from verl.utils.tsc.clustering import cluster_by_tool_behavior
        clusters = cluster_by_tool_behavior(candidates, tool_results)

        config = TSCConfig(normalize_rewards=False)
        rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

        # Two correct solutions should get high rewards
        assert rewards[0] > rewards[2]
        assert rewards[1] > rewards[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python
"""
Basic integration test for TSC functionality.
This script tests TSC components without requiring pytest or full VERL dependencies.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def test_clustering():
    """Test basic clustering functionality."""
    print("\n=== Testing Clustering ===")

    # Import directly from module files to avoid verl.__init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "clustering",
        "verl/utils/tsc/clustering.py"
    )
    clustering = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clustering)

    normalize_text = clustering.normalize_text
    extract_numeric_answer = clustering.extract_numeric_answer
    cluster_candidates = clustering.cluster_candidates

    # Test text normalization
    assert normalize_text("  Hello   World  ") == "hello world"
    print("✓ Text normalization works")

    # Test numeric extraction
    assert extract_numeric_answer("42") == 42.0
    assert extract_numeric_answer("The answer is 42") == 42.0
    print("✓ Numeric extraction works")

    # Test clustering
    candidates = ["4", "4", "5"]
    tool_results = [
        {'score': 1.0, 'passed': True},
        {'score': 1.0, 'passed': True},
        {'score': 0.0, 'passed': False},
    ]

    clusters = cluster_candidates(candidates, tool_results, mode='numeric')
    assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
    assert clusters[0].size == 2, f"Expected first cluster size 2, got {clusters[0].size}"
    print(f"✓ Clustering works: {len(clusters)} clusters, sizes: {[c.size for c in clusters]}")


def test_reward_computation():
    """Test reward computation."""
    print("\n=== Testing Reward Computation ===")

    from verl.utils.tsc.reward_computation import (
        TSCConfig,
        compute_tsc_rewards,
        compute_tsc_metrics,
    )
    from verl.utils.tsc.clustering import Cluster

    # Create test scenario: 3 correct, 1 wrong
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

    # Compute rewards
    config = TSCConfig(normalize_rewards=False)
    rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

    assert len(rewards) == 4, f"Expected 4 rewards, got {len(rewards)}"
    assert all(rewards[i] > rewards[3] for i in range(3)), \
        f"Correct majority should have higher rewards than wrong answer"
    print(f"✓ Reward computation works: rewards = {rewards}")

    # Compute metrics
    metrics = compute_tsc_metrics(candidates, tool_results, clusters, rewards, config)
    assert 'avg_tool_score' in metrics
    assert 'num_clusters' in metrics
    assert metrics['num_clusters'] == 2
    print(f"✓ Metrics computation works: {len(metrics)} metrics computed")


def test_tool_evaluation():
    """Test tool evaluation (basic, without external dependencies)."""
    print("\n=== Testing Tool Evaluation ===")

    from verl.utils.tsc.tool_evaluation import (
        ToolEvaluationResult,
    )

    # Test ToolEvaluationResult
    result = ToolEvaluationResult(score=1.0, passed=True)
    assert result.score == 1.0
    assert result.passed is True
    print("✓ ToolEvaluationResult works")

    d = result.to_dict()
    assert d['score'] == 1.0
    assert d['passed'] is True
    print("✓ ToolEvaluationResult.to_dict() works")


def test_config():
    """Test TSC configuration."""
    print("\n=== Testing TSC Config ===")

    from verl.utils.tsc.reward_computation import TSCConfig

    # Default config
    config = TSCConfig()
    assert config.alpha_tool == 0.8
    assert config.beta_consistency == 0.2
    config.validate()  # Should not raise
    print("✓ Default config valid")

    # Custom config
    config = TSCConfig(
        alpha_tool=0.9,
        beta_consistency=0.1,
        correct_majority_bonus=0.2,
    )
    config.validate()
    assert config.alpha_tool == 0.9
    print("✓ Custom config valid")

    # Invalid config
    try:
        bad_config = TSCConfig(alpha_tool=1.5)
        bad_config.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Config validation catches invalid values")


def test_integration_scenario():
    """Test a complete TSC scenario."""
    print("\n=== Testing Complete TSC Scenario ===")

    from verl.utils.tsc import (
        cluster_candidates,
        compute_tsc_rewards,
        TSCConfig,
    )

    # Scenario: Math problem "2+2", 5 candidates
    # 3 say "4", 2 say "5"
    candidates = ["4", "4.0", "The answer is 4", "5", "5.0"]
    tool_results = [
        {'score': 1.0, 'passed': True},
        {'score': 1.0, 'passed': True},
        {'score': 1.0, 'passed': True},
        {'score': 0.0, 'passed': False},
        {'score': 0.0, 'passed': False},
    ]

    # Cluster candidates
    clusters = cluster_candidates(candidates, tool_results, mode='numeric')
    print(f"  Clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"    Cluster {i}: size={cluster.size}, correct={cluster.is_tool_correct}")

    # Compute rewards
    config = TSCConfig(
        alpha_tool=0.8,
        beta_consistency=0.2,
        normalize_rewards=False,
    )
    rewards = compute_tsc_rewards(candidates, tool_results, clusters, config)

    print(f"  Rewards: {[f'{r:.3f}' for r in rewards]}")

    # Verify correct majority gets higher rewards
    correct_indices = [0, 1, 2]
    wrong_indices = [3, 4]

    avg_correct = sum(rewards[i] for i in correct_indices) / len(correct_indices)
    avg_wrong = sum(rewards[i] for i in wrong_indices) / len(wrong_indices)

    assert avg_correct > avg_wrong, \
        f"Correct answers should have higher avg reward ({avg_correct:.3f} vs {avg_wrong:.3f})"

    print(f"✓ Integration scenario works: correct avg={avg_correct:.3f}, wrong avg={avg_wrong:.3f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TSC Basic Integration Tests")
    print("=" * 60)

    tests = [
        test_clustering,
        test_reward_computation,
        test_tool_evaluation,
        test_config,
        test_integration_scenario,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if not failed:
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(failed)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()

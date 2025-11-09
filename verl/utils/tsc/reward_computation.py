"""
Reward computation for tool-grounded self-consistency (TSC).

This module implements the core TSC reward logic that combines:
1. Tool-based correctness scores (from unit tests, calculators, etc.)
2. Self-consistency scores (agreement among candidate solutions)

The reward design ensures:
- Agreement only helps when tied to correctness
- Wrong majorities are penalized
- Correct minority dissent is allowed/rewarded
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from .clustering import Cluster

logger = logging.getLogger(__name__)


@dataclass
class TSCConfig:
    """Configuration for tool-grounded self-consistency rewards.

    This configuration controls how rewards are computed from tool scores
    and self-consistency clusters.
    """

    # Core weights
    alpha_tool: float = 0.8
    """Weight for tool correctness component (default: 0.8)"""

    beta_consistency: float = 0.2
    """Weight for self-consistency component (default: 0.2)"""

    # Bonuses and penalties
    correct_majority_bonus: float = 0.1
    """Bonus for being in correct majority cluster (default: 0.1)"""

    wrong_majority_penalty: float = 0.3
    """Penalty for being in wrong majority cluster (default: 0.3)"""

    correct_minority_bonus: float = 0.15
    """Bonus for correct minority when majority is wrong (default: 0.15)"""

    # Thresholds
    min_cluster_frac_for_majority: float = 0.5
    """Minimum fraction to be considered majority (default: 0.5)"""

    tool_correct_threshold: float = 0.8
    """Tool score threshold to consider cluster correct (default: 0.8)"""

    # Behavior flags
    allow_correct_minority_reward: bool = True
    """Whether to reward correct minorities (default: True)"""

    normalize_rewards: bool = True
    """Whether to normalize rewards to zero mean (default: True)"""

    reward_scale: float = 1.0
    """Global scaling factor for rewards (default: 1.0)"""

    # Optional reward clipping
    reward_clip_min: Optional[float] = None
    """Minimum reward value (None = no clipping)"""

    reward_clip_max: Optional[float] = None
    """Maximum reward value (None = no clipping)"""

    def validate(self):
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if not (0.0 <= self.alpha_tool <= 1.0):
            raise ValueError(f"alpha_tool must be in [0, 1], got {self.alpha_tool}")

        if not (0.0 <= self.beta_consistency <= 1.0):
            raise ValueError(f"beta_consistency must be in [0, 1], got {self.beta_consistency}")

        if not (0.0 < self.min_cluster_frac_for_majority <= 1.0):
            raise ValueError(
                f"min_cluster_frac_for_majority must be in (0, 1], "
                f"got {self.min_cluster_frac_for_majority}"
            )

        if not (0.0 <= self.tool_correct_threshold <= 1.0):
            raise ValueError(
                f"tool_correct_threshold must be in [0, 1], "
                f"got {self.tool_correct_threshold}"
            )

        if self.reward_clip_min is not None and self.reward_clip_max is not None:
            if self.reward_clip_min >= self.reward_clip_max:
                raise ValueError(
                    f"reward_clip_min ({self.reward_clip_min}) must be < "
                    f"reward_clip_max ({self.reward_clip_max})"
                )


def identify_best_cluster(clusters: List[Cluster]) -> Cluster:
    """Identify the best cluster based on tool correctness and size.

    The best cluster is determined by:
    1. Highest average tool score
    2. Tie-break by cluster size

    Args:
        clusters: List of Cluster objects

    Returns:
        The best cluster

    Raises:
        ValueError: If clusters list is empty
    """
    if not clusters:
        raise ValueError("Cannot identify best cluster from empty list")

    # Sort by avg_tool_score (descending), then by size (descending)
    best_cluster = max(
        clusters,
        key=lambda c: (c.avg_tool_score, c.size)
    )

    return best_cluster


def is_majority_cluster(cluster: Cluster, total_candidates: int, threshold: float) -> bool:
    """Check if a cluster represents a majority.

    Args:
        cluster: Cluster to check
        total_candidates: Total number of candidates
        threshold: Minimum fraction to be considered majority

    Returns:
        True if cluster is a majority
    """
    fraction = cluster.fraction_of_total(total_candidates)
    return fraction >= threshold


def is_cluster_correct(cluster: Cluster, threshold: float) -> bool:
    """Check if a cluster is considered tool-correct.

    Args:
        cluster: Cluster to check
        threshold: Tool score threshold for correctness

    Returns:
        True if cluster is correct
    """
    return cluster.avg_tool_score >= threshold


def compute_tsc_rewards(
    candidates: List[str],
    tool_results: List[Dict[str, Any]],
    clusters: List[Cluster],
    config: TSCConfig,
) -> List[float]:
    """Compute tool-grounded self-consistency rewards.

    This is the core TSC reward computation function. For each candidate,
    it computes a reward based on:
    1. Tool correctness score
    2. Self-consistency (cluster membership)
    3. Bonuses/penalties based on majority/minority status

    Reward logic:
    - Correct majority: high reward (tool + consistency + bonus)
    - Wrong majority: penalty (reduced reward)
    - Correct minority (when majority wrong): rewarded if enabled
    - Other cases: primarily tool-based reward

    Args:
        candidates: List of candidate solution strings
        tool_results: List of tool evaluation result dictionaries
        clusters: List of Cluster objects (from clustering.py)
        config: TSCConfig with reward parameters

    Returns:
        List of reward values (one per candidate)

    Raises:
        ValueError: If inputs have inconsistent lengths or config is invalid
    """
    # Validate inputs
    if len(candidates) != len(tool_results):
        raise ValueError(
            f"candidates ({len(candidates)}) and tool_results ({len(tool_results)}) "
            f"must have same length"
        )

    config.validate()

    num_candidates = len(candidates)

    if num_candidates == 0:
        return []

    # Build index-to-cluster mapping
    idx_to_cluster: Dict[int, Cluster] = {}
    for cluster in clusters:
        for idx in cluster.members:
            idx_to_cluster[idx] = cluster

    # Identify best cluster
    best_cluster = identify_best_cluster(clusters)

    # Identify majority cluster (if any)
    majority_cluster = None
    for cluster in clusters:
        if is_majority_cluster(cluster, num_candidates, config.min_cluster_frac_for_majority):
            # If multiple clusters could be majority, take the largest
            if majority_cluster is None or cluster.size > majority_cluster.size:
                majority_cluster = cluster

    # Check if majority is correct
    majority_is_correct = False
    if majority_cluster is not None:
        majority_is_correct = is_cluster_correct(majority_cluster, config.tool_correct_threshold)

    # Compute rewards for each candidate
    rewards = []

    for idx in range(num_candidates):
        # Get tool score for this candidate
        tool_score = tool_results[idx].get('score', 0.0)

        # Get cluster for this candidate
        cluster = idx_to_cluster.get(idx)

        if cluster is None:
            # This shouldn't happen if clustering worked correctly
            logger.warning(f"Candidate {idx} not in any cluster, using tool score only")
            reward = config.alpha_tool * tool_score
            rewards.append(reward)
            continue

        # Compute base tool reward
        r_tool = tool_score

        # Compute consistency component
        frac_c = cluster.fraction_of_total(num_candidates)
        r_consistency = frac_c

        # Check cluster status
        is_in_majority = (majority_cluster is not None and cluster == majority_cluster)
        is_cluster_tool_correct = is_cluster_correct(cluster, config.tool_correct_threshold)

        # Compute final reward based on cluster status
        if is_in_majority:
            if majority_is_correct:
                # Correct majority: reward agreement + tool correctness
                reward = (
                    config.alpha_tool * r_tool +
                    config.beta_consistency * r_consistency +
                    config.correct_majority_bonus
                )
            else:
                # Wrong majority: penalize
                reward = (
                    config.alpha_tool * r_tool -
                    config.wrong_majority_penalty
                )
        else:
            # Not in majority cluster
            if (config.allow_correct_minority_reward and
                is_cluster_tool_correct and
                not majority_is_correct):
                # Correct minority when majority is wrong: reward dissent
                reward = (
                    config.alpha_tool * r_tool +
                    config.correct_minority_bonus
                )
            else:
                # Neutral case: primarily tool-based
                reward = config.alpha_tool * r_tool

        rewards.append(reward)

    # Convert to numpy array for normalization
    rewards = np.array(rewards, dtype=np.float32)

    # Normalize rewards (zero mean, preserve std)
    if config.normalize_rewards and num_candidates > 1:
        mean_reward = np.mean(rewards)
        rewards = rewards - mean_reward

    # Apply global scaling
    rewards = rewards * config.reward_scale

    # Apply clipping if configured
    if config.reward_clip_min is not None:
        rewards = np.maximum(rewards, config.reward_clip_min)
    if config.reward_clip_max is not None:
        rewards = np.minimum(rewards, config.reward_clip_max)

    return rewards.tolist()


def compute_tsc_metrics(
    candidates: List[str],
    tool_results: List[Dict[str, Any]],
    clusters: List[Cluster],
    rewards: List[float],
    config: TSCConfig,
) -> Dict[str, float]:
    """Compute diagnostic metrics for TSC rewards.

    These metrics are useful for monitoring and debugging the TSC system.

    Args:
        candidates: List of candidate solution strings
        tool_results: List of tool evaluation result dictionaries
        clusters: List of Cluster objects
        rewards: Computed rewards
        config: TSCConfig

    Returns:
        Dictionary of metric name -> value
    """
    num_candidates = len(candidates)

    if num_candidates == 0:
        return {}

    metrics = {}

    # Average tool score
    tool_scores = [tr.get('score', 0.0) for tr in tool_results]
    metrics['avg_tool_score'] = float(np.mean(tool_scores))
    metrics['std_tool_score'] = float(np.std(tool_scores))

    # Cluster statistics
    metrics['num_clusters'] = len(clusters)

    if len(clusters) > 0:
        cluster_sizes = [c.size for c in clusters]
        metrics['avg_cluster_size'] = float(np.mean(cluster_sizes))
        metrics['max_cluster_size'] = max(cluster_sizes)
        metrics['min_cluster_size'] = min(cluster_sizes)

        # Cluster entropy (diversity measure)
        cluster_fractions = [c.size / num_candidates for c in clusters]
        entropy = -sum(p * np.log(p + 1e-10) for p in cluster_fractions if p > 0)
        metrics['cluster_entropy'] = float(entropy)

        # Identify majority and best clusters
        best_cluster = identify_best_cluster(clusters)
        metrics['best_cluster_size'] = best_cluster.size
        metrics['best_cluster_avg_tool_score'] = best_cluster.avg_tool_score

        majority_cluster = None
        for cluster in clusters:
            if is_majority_cluster(cluster, num_candidates, config.min_cluster_frac_for_majority):
                if majority_cluster is None or cluster.size > majority_cluster.size:
                    majority_cluster = cluster

        if majority_cluster is not None:
            metrics['has_majority'] = 1.0
            metrics['majority_cluster_size'] = majority_cluster.size
            metrics['majority_cluster_avg_tool_score'] = majority_cluster.avg_tool_score

            # Check if majority is correct
            majority_is_correct = is_cluster_correct(
                majority_cluster, config.tool_correct_threshold
            )
            metrics['majority_is_correct'] = float(majority_is_correct)

            # Count candidates in correct/wrong majority
            if majority_is_correct:
                metrics['frac_correct_majority'] = majority_cluster.size / num_candidates
                metrics['frac_wrong_majority'] = 0.0
            else:
                metrics['frac_correct_majority'] = 0.0
                metrics['frac_wrong_majority'] = majority_cluster.size / num_candidates

            # Count correct minorities
            correct_minority_count = 0
            for cluster in clusters:
                if (cluster != majority_cluster and
                    is_cluster_correct(cluster, config.tool_correct_threshold)):
                    correct_minority_count += cluster.size

            metrics['frac_correct_minority'] = correct_minority_count / num_candidates
        else:
            metrics['has_majority'] = 0.0
            metrics['frac_correct_majority'] = 0.0
            metrics['frac_wrong_majority'] = 0.0
            metrics['frac_correct_minority'] = 0.0

    # Reward statistics
    metrics['avg_reward'] = float(np.mean(rewards))
    metrics['std_reward'] = float(np.std(rewards))
    metrics['min_reward'] = float(np.min(rewards))
    metrics['max_reward'] = float(np.max(rewards))

    # Consistency score (avg cluster fraction weighted by cluster size)
    if len(clusters) > 0:
        weighted_consistency = sum(
            c.size * c.fraction_of_total(num_candidates) for c in clusters
        ) / num_candidates
        metrics['avg_consistency_score'] = float(weighted_consistency)

    return metrics

"""
Tool-Grounded Self-Consistency (TSC) Reward Manager for VERL.

This reward manager implements TSC rewards by:
1. Grouping multiple candidate solutions per problem (by UID)
2. Evaluating each candidate with external tools
3. Clustering candidates by semantic equivalence
4. Computing rewards based on both tool correctness and self-consistency
"""

from collections import defaultdict
import logging
from typing import Dict, Any, Optional, List

import torch
import numpy as np

from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.tsc import (
    cluster_candidates,
    evaluate_with_tools,
    create_evaluator,
    compute_tsc_rewards,
    compute_tsc_metrics,
    TSCConfig,
    ToolEvaluationResult,
)

logger = logging.getLogger(__name__)


@register("tsc")
class TSCRewardManager:
    """Tool-Grounded Self-Consistency Reward Manager.

    This reward manager computes rewards based on both:
    - Tool evaluation (unit tests, calculators, constraint checkers)
    - Self-consistency (agreement among candidate solutions)

    The reward design ensures:
    - Agreement helps only when tied to correctness
    - Wrong majorities are penalized
    - Correct minority dissent is rewarded
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        # TSC-specific parameters
        tool_type: str = "auto",
        clustering_mode: str = "auto",
        # TSC config parameters
        alpha_tool: float = 0.8,
        beta_consistency: float = 0.2,
        correct_majority_bonus: float = 0.1,
        wrong_majority_penalty: float = 0.3,
        correct_minority_bonus: float = 0.15,
        min_cluster_frac_for_majority: float = 0.5,
        tool_correct_threshold: float = 0.8,
        allow_correct_minority_reward: bool = True,
        normalize_rewards: bool = True,
        reward_scale: float = 1.0,
        # Tool evaluator parameters
        sandbox_fusion_url: Optional[str] = None,
        concurrent_semaphore: Optional[Any] = None,
        code_timeout: float = 10.0,
        # Other parameters
        reward_fn_key: str = "data_source",
        debug: bool = False,
    ) -> None:
        """Initialize TSC Reward Manager.

        Args:
            tokenizer: Tokenizer for decoding token IDs
            num_examine: Number of batches to print for debugging
            tool_type: Type of tool ("auto", "math", "code", "qa", "custom")
            clustering_mode: Clustering mode ("auto", "tool", "text", "numeric")
            alpha_tool: Weight for tool correctness (default: 0.8)
            beta_consistency: Weight for self-consistency (default: 0.2)
            correct_majority_bonus: Bonus for correct majority (default: 0.1)
            wrong_majority_penalty: Penalty for wrong majority (default: 0.3)
            correct_minority_bonus: Bonus for correct minority (default: 0.15)
            min_cluster_frac_for_majority: Min fraction for majority (default: 0.5)
            tool_correct_threshold: Tool score threshold for correctness (default: 0.8)
            allow_correct_minority_reward: Whether to reward correct minorities (default: True)
            normalize_rewards: Whether to normalize rewards (default: True)
            reward_scale: Global scaling factor (default: 1.0)
            sandbox_fusion_url: URL for code sandbox (optional)
            concurrent_semaphore: Semaphore for concurrent execution (optional)
            code_timeout: Timeout for code execution (default: 10.0)
            reward_fn_key: Key for data source in non_tensor_batch (default: "data_source")
            debug: Enable debug logging (default: False)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.tool_type = tool_type
        self.clustering_mode = clustering_mode
        self.reward_fn_key = reward_fn_key
        self.debug = debug

        # Code execution parameters
        self.sandbox_fusion_url = sandbox_fusion_url
        self.concurrent_semaphore = concurrent_semaphore
        self.code_timeout = code_timeout

        # Create TSC configuration
        self.tsc_config = TSCConfig(
            alpha_tool=alpha_tool,
            beta_consistency=beta_consistency,
            correct_majority_bonus=correct_majority_bonus,
            wrong_majority_penalty=wrong_majority_penalty,
            correct_minority_bonus=correct_minority_bonus,
            min_cluster_frac_for_majority=min_cluster_frac_for_majority,
            tool_correct_threshold=tool_correct_threshold,
            allow_correct_minority_reward=allow_correct_minority_reward,
            normalize_rewards=normalize_rewards,
            reward_scale=reward_scale,
        )

        # Validate config
        self.tsc_config.validate()

        # Statistics
        self.already_printed = {}
        self.call_count = 0

    def _create_evaluator_for_data_source(self, data_source: str):
        """Create appropriate evaluator based on data source.

        Args:
            data_source: Dataset identifier

        Returns:
            ToolEvaluator instance
        """
        # Auto-detect tool type if needed
        tool_type = self.tool_type
        if tool_type == "auto":
            if "gsm8k" in data_source.lower() or "math" in data_source.lower():
                tool_type = "math"
            elif any(x in data_source.lower() for x in ["code", "apps", "taco"]):
                tool_type = "code"
            elif "search" in data_source.lower():
                tool_type = "qa"
            else:
                # Default to math for unknown datasets
                logger.warning(f"Unknown data_source {data_source}, defaulting to math evaluator")
                tool_type = "math"

        # Create evaluator
        if tool_type == "code":
            return create_evaluator(
                tool_type="code",
                data_source=data_source,
                sandbox_fusion_url=self.sandbox_fusion_url,
                concurrent_semaphore=self.concurrent_semaphore,
                timeout=self.code_timeout,
                continuous=True,  # Use continuous scoring for partial credit
            )
        else:
            return create_evaluator(
                tool_type=tool_type,
                data_source=data_source,
            )

    def __call__(self, data: DataProto, return_dict=False):
        """Compute TSC rewards for a batch of data.

        Args:
            data: DataProto containing batch of prompts and responses
            return_dict: If True, return dict with reward_tensor and extra_info

        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        self.call_count += 1

        # Initialize reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Group candidates by UID (problem ID)
        uid_to_indices: Dict[str, List[int]] = defaultdict(list)
        uid_to_data_source: Dict[str, str] = {}

        for i in range(len(data)):
            data_item = data[i]
            uid = data_item.non_tensor_batch.get("uid", f"default_{i}")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")

            uid_to_indices[uid].append(i)
            if uid not in uid_to_data_source:
                uid_to_data_source[uid] = data_source

        # Process each problem (UID group)
        all_tsc_metrics = []

        for uid, indices in uid_to_indices.items():
            if len(indices) < 2:
                logger.warning(
                    f"TSC requires at least 2 candidates per problem. "
                    f"UID {uid} has only {len(indices)} candidates. "
                    f"Using simple tool-based reward."
                )

            data_source = uid_to_data_source[uid]

            # Extract candidates and ground truth
            candidates = []
            ground_truths = []
            extra_infos = []
            valid_response_lengths = []

            for idx in indices:
                data_item = data[idx]

                # Extract response
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                candidates.append(response_str)
                valid_response_lengths.append(valid_response_length)

                # Extract ground truth and extra info
                gt = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                ground_truths.append(gt)

                extra_info = data_item.non_tensor_batch.get("extra_info", None)
                extra_infos.append(extra_info)

            # Create evaluator for this data source
            try:
                evaluator = self._create_evaluator_for_data_source(data_source)
            except Exception as e:
                logger.error(f"Failed to create evaluator for {data_source}: {e}")
                # Fallback: assign zero rewards
                for idx, valid_len in zip(indices, valid_response_lengths):
                    reward_tensor[idx, valid_len - 1] = 0.0
                continue

            # Evaluate all candidates with tools
            tool_results = []
            for candidate, gt, extra_info in zip(candidates, ground_truths, extra_infos):
                try:
                    result = evaluate_with_tools(
                        evaluator=evaluator,
                        problem="",  # Not used by most evaluators
                        candidate=candidate,
                        ground_truth=gt,
                        extra_info=extra_info,
                        safe_fallback=True,
                    )
                    tool_results.append(result.to_dict())
                except Exception as e:
                    logger.error(f"Tool evaluation failed for UID {uid}: {e}")
                    # Safe fallback
                    tool_results.append({
                        'score': 0.0,
                        'passed': False,
                        'error': str(e),
                    })

            # Cluster candidates
            try:
                clusters = cluster_candidates(
                    candidates=candidates,
                    tool_results=tool_results,
                    mode=self.clustering_mode,
                )
            except Exception as e:
                logger.error(f"Clustering failed for UID {uid}: {e}")
                # Fallback: assign tool-based rewards only
                for idx, tool_result, valid_len in zip(indices, tool_results, valid_response_lengths):
                    reward_tensor[idx, valid_len - 1] = tool_result['score']
                continue

            # Compute TSC rewards
            if len(indices) >= 2 and len(clusters) > 0:
                try:
                    rewards = compute_tsc_rewards(
                        candidates=candidates,
                        tool_results=tool_results,
                        clusters=clusters,
                        config=self.tsc_config,
                    )
                except Exception as e:
                    logger.error(f"TSC reward computation failed for UID {uid}: {e}")
                    # Fallback: assign tool-based rewards only
                    rewards = [tr['score'] for tr in tool_results]
            else:
                # Not enough candidates for TSC, use tool scores directly
                rewards = [tr['score'] for tr in tool_results]

            # Assign rewards to tensor
            for idx, reward, valid_len in zip(indices, rewards, valid_response_lengths):
                reward_tensor[idx, valid_len - 1] = reward

            # Compute and log metrics
            if len(indices) >= 2 and len(clusters) > 0:
                try:
                    metrics = compute_tsc_metrics(
                        candidates=candidates,
                        tool_results=tool_results,
                        clusters=clusters,
                        rewards=rewards,
                        config=self.tsc_config,
                    )
                    all_tsc_metrics.append(metrics)

                    # Store metrics in extra_info
                    for key, value in metrics.items():
                        reward_extra_info[f'tsc/{key}'].append(value)

                except Exception as e:
                    logger.error(f"TSC metrics computation failed for UID {uid}: {e}")

            # Debug printing
            if data_source not in self.already_printed:
                self.already_printed[data_source] = 0

            if self.already_printed[data_source] < self.num_examine:
                self.already_printed[data_source] += 1
                print(f"\n[TSC Debug - UID: {uid}, Data Source: {data_source}]")
                print(f"Number of candidates: {len(candidates)}")
                print(f"Number of clusters: {len(clusters)}")

                for i, (cand, tool_res, rew) in enumerate(zip(candidates, tool_results, rewards)):
                    print(f"\nCandidate {i}:")
                    print(f"  Response: {cand[:200]}...")
                    print(f"  Tool score: {tool_res['score']:.3f}")
                    print(f"  Tool passed: {tool_res['passed']}")
                    print(f"  TSC reward: {rew:.3f}")

                if len(clusters) > 0:
                    print("\nClusters:")
                    for j, cluster in enumerate(clusters):
                        print(f"  Cluster {j}: size={cluster.size}, "
                              f"avg_tool_score={cluster.avg_tool_score:.3f}, "
                              f"is_correct={cluster.is_tool_correct}")

        # Aggregate and log TSC metrics
        if all_tsc_metrics and self.debug:
            aggregated_metrics = {}
            for key in all_tsc_metrics[0].keys():
                values = [m[key] for m in all_tsc_metrics if key in m]
                if values:
                    aggregated_metrics[key] = np.mean(values)

            logger.info(f"TSC Metrics (call {self.call_count}): {aggregated_metrics}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for integrating model-based voting with reward managers.

This module provides helper functions to create generation callbacks that
can be passed to voting-based reward managers like PeerReviewVotingRewardManager.
"""

import torch
from typing import Callable, Optional


def create_simple_generation_fn(model, tokenizer, device='cuda') -> Callable:
    """
    Create a simple generation function from a model and tokenizer.

    This function can be passed to reward managers that support model-based voting.

    Args:
        model: The language model (e.g., actor model)
        tokenizer: The tokenizer for the model
        device: Device to run generation on

    Returns:
        A callable that takes (prompt, max_new_tokens, temperature) and returns generated text

    Example:
        >>> generation_fn = create_simple_generation_fn(actor_model, tokenizer)
        >>> reward_manager = PeerReviewVotingRewardManager(
        ...     tokenizer=tokenizer,
        ...     num_examine=10,
        ...     use_model_voting=True,
        ...     generation_fn=generation_fn
        ... )
    """

    def generation_fn(prompt: str, max_new_tokens: int = 10, temperature: float = 0.7) -> str:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    return generation_fn


def create_batched_generation_fn(model, tokenizer, device='cuda', batch_size: int = 8) -> Callable:
    """
    Create a batched generation function for more efficient voting.

    This is useful when you have many votes to generate and want to batch them.

    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        device: Device to run generation on
        batch_size: Batch size for generation

    Returns:
        A callable that takes (prompt, max_new_tokens, temperature) and returns generated text

    Note:
        This function still returns a single string, but internally it could be
        optimized to batch multiple voting requests together.
    """

    def generation_fn(prompt: str, max_new_tokens: int = 10, temperature: float = 0.7) -> str:
        """
        Generate text using the model with batching support.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # For now, this is the same as simple generation
        # In a full implementation, you might accumulate prompts and batch them
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    return generation_fn


def create_actor_wrapper_generation_fn(actor_worker_group, tokenizer) -> Callable:
    """
    Create a generation function that uses the actor worker group from the trainer.

    This is for integration with the Ray-based distributed training setup.

    Args:
        actor_worker_group: The actor worker group from the trainer
        tokenizer: The tokenizer

    Returns:
        A callable generation function

    Example:
        >>> # In the trainer code:
        >>> generation_fn = create_actor_wrapper_generation_fn(
        ...     self.actor_rollout_wg,
        ...     self.tokenizer
        ... )
        >>> reward_manager = PeerReviewVotingRewardManager(
        ...     tokenizer=self.tokenizer,
        ...     num_examine=10,
        ...     use_model_voting=True,
        ...     generation_fn=generation_fn
        ... )
    """

    def generation_fn(prompt: str, max_new_tokens: int = 10, temperature: float = 0.7) -> str:
        """
        Generate text using the actor worker group.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # This is a placeholder - actual implementation would need to:
        # 1. Create a DataProto with the prompt
        # 2. Call actor_worker_group.generate_sequences()
        # 3. Extract and return the generated text

        # For now, raise NotImplementedError with instructions
        raise NotImplementedError(
            "Actor wrapper generation is not yet fully implemented. "
            "To use model-based voting with the distributed training setup, "
            "you need to implement the actor_worker_group integration. "
            "For simpler setups, use create_simple_generation_fn instead."
        )

    return generation_fn


class VotingCache:
    """
    Cache for voting results to avoid redundant model calls.

    When using model-based voting, the same set of answers might appear
    multiple times. This cache stores voting results to avoid redundant
    generation calls.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the voting cache.

        Args:
            max_size: Maximum number of entries to cache
        """
        self.cache = {}
        self.max_size = max_size

    def get_cache_key(self, prompt: str, answers: list) -> str:
        """
        Generate a cache key from prompt and answers.

        Args:
            prompt: The voting prompt
            answers: List of candidate answers

        Returns:
            Cache key string
        """
        # Create a deterministic key from prompt and sorted answers
        sorted_answers = tuple(sorted(str(a) for a in answers if a is not None))
        return f"{hash(prompt)}:{hash(sorted_answers)}"

    def get(self, prompt: str, answers: list) -> Optional[dict]:
        """
        Get cached voting results.

        Args:
            prompt: The voting prompt
            answers: List of candidate answers

        Returns:
            Cached vote counts or None if not in cache
        """
        key = self.get_cache_key(prompt, answers)
        return self.cache.get(key)

    def put(self, prompt: str, answers: list, vote_counts: dict):
        """
        Store voting results in cache.

        Args:
            prompt: The voting prompt
            answers: List of candidate answers
            vote_counts: Dictionary of vote counts to cache
        """
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self.get_cache_key(prompt, answers)
        self.cache[key] = vote_counts

    def clear(self):
        """Clear the cache."""
        self.cache.clear()

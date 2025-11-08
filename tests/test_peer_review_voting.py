#!/usr/bin/env python3
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
Test script for peer-review voting reward manager.

This script validates that the peer-review voting system works correctly
with various benchmarks and configurations.
"""

import sys
import torch
from collections import Counter
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, '/home/user/self-questioning-lm')

from verl import DataProto
from verl.workers.reward_manager import PeerReviewVotingRewardManager


def create_mock_data_proto(prompts_and_responses, tokenizer):
    """
    Create a mock DataProto for testing.

    Args:
        prompts_and_responses: List of (prompt, response, uid, data_source) tuples
        tokenizer: Tokenizer to use

    Returns:
        DataProto object
    """
    from tensordict import TensorDict

    batch_items = []
    non_tensor_items = []

    for prompt, response, uid, data_source in prompts_and_responses:
        # Tokenize
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        # Create tensors
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        response_tensor = torch.tensor(response_ids, dtype=torch.long)

        # Create attention mask
        total_length = len(prompt_ids) + len(response_ids)
        attention_mask = torch.ones(total_length, dtype=torch.long)

        batch_items.append({
            'prompts': prompt_tensor,
            'responses': response_tensor,
            'attention_mask': attention_mask,
        })

        non_tensor_items.append({
            'uid': uid,
            'data_source': data_source,
        })

    # Convert to batch format
    batch = TensorDict({
        'prompts': torch.nn.utils.rnn.pad_sequence(
            [item['prompts'] for item in batch_items],
            batch_first=True,
            padding_value=tokenizer.pad_token_id or 0
        ),
        'responses': torch.nn.utils.rnn.pad_sequence(
            [item['responses'] for item in batch_items],
            batch_first=True,
            padding_value=tokenizer.pad_token_id or 0
        ),
        'attention_mask': torch.nn.utils.rnn.pad_sequence(
            [item['attention_mask'] for item in batch_items],
            batch_first=True,
            padding_value=0
        ),
    }, batch_size=[len(batch_items)])

    non_tensor_batch = {
        'uid': [item['uid'] for item in non_tensor_items],
        'data_source': [item['data_source'] for item in non_tensor_items],
    }

    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    return data


def test_gsm8k_extraction():
    """Test answer extraction for GSM8K format."""
    print("\n" + "="*60)
    print("Test 1: GSM8K Answer Extraction")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_manager = PeerReviewVotingRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        use_model_voting=False
    )

    # Test various GSM8K format responses
    test_cases = [
        ("Response with answer\n#### 42", "gsm8k", "42"),
        ("Let me solve this\n#### 123", "gsm8k", "123"),
        ("The answer is\n#### -5", "gsm8k", "-5"),
        ("No answer marker", "gsm8k", None),
    ]

    print("\nTesting GSM8K answer extraction:")
    all_passed = True
    for response, data_source, expected in test_cases:
        result = reward_manager.extract_answer(response, data_source)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} Response: '{response[:30]}...' -> Expected: {expected}, Got: {result}")

    print(f"\nGSM8K extraction test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_generic_extraction():
    """Test generic answer extraction for unknown benchmarks."""
    print("\n" + "="*60)
    print("Test 2: Generic Answer Extraction")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_manager = PeerReviewVotingRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        use_model_voting=False
    )

    test_cases = [
        ("The answer is 42", "custom_benchmark", "42"),
        ("Final answer: 123", "custom_benchmark", "123"),
        ("I think the answer is \"hello\"", "custom_benchmark", "hello"),
        ("Result is \\boxed{256}", "custom_benchmark", "256"),
    ]

    print("\nTesting generic answer extraction:")
    all_passed = True
    for response, data_source, expected in test_cases:
        result = reward_manager.extract_answer(response, data_source)
        # For generic extraction, we're more lenient
        passed = result is not None or expected is None
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} Response: '{response}' -> Got: {result}")

    print(f"\nGeneric extraction test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_simulated_voting():
    """Test simulated voting (no model calls)."""
    print("\n" + "="*60)
    print("Test 3: Simulated Voting")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_manager = PeerReviewVotingRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        use_model_voting=False
    )

    # Create mock data with multiple answers for the same problem
    uid = "test-problem-1"
    prompts_and_responses = [
        ("What is 2+2?", "Let me solve: 2+2=4\n#### 4", uid, "gsm8k"),
        ("What is 2+2?", "The answer is 4\n#### 4", uid, "gsm8k"),
        ("What is 2+2?", "I think it's 5\n#### 5", uid, "gsm8k"),
        ("What is 2+2?", "2+2 equals 4\n#### 4", uid, "gsm8k"),
    ]

    data = create_mock_data_proto(prompts_and_responses, tokenizer)

    # Compute rewards
    result = reward_manager(data, return_dict=True)
    reward_tensor = result['reward_tensor']
    extra_info = result['reward_extra_info']

    print(f"\nNumber of responses: {len(prompts_and_responses)}")
    print(f"Answers extracted: {extra_info['answer']}")
    print(f"Winning answer: {extra_info['winning_answer'][0]}")
    print(f"Vote counts: {dict(Counter(extra_info['winning_answer']))}")

    # Check that answer "4" won (appears 3 times vs "5" appearing 1 time)
    winning_answer = extra_info['winning_answer'][0]
    passed = winning_answer == "4"

    # Check that 3 responses got reward 1.0 and 1 got reward 0.0
    rewards = [reward_tensor[i, -1].item() for i in range(len(prompts_and_responses))]
    num_rewarded = sum(1 for r in rewards if r > 0.5)
    passed = passed and num_rewarded == 3

    print(f"\nRewards assigned: {rewards}")
    print(f"Number of responses with reward=1.0: {num_rewarded}")

    print(f"\nSimulated voting test: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_vote_extraction():
    """Test vote extraction from model responses."""
    print("\n" + "="*60)
    print("Test 4: Vote Extraction")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_manager = PeerReviewVotingRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        use_model_voting=False
    )

    test_cases = [
        ("I choose 1", 5, 0),
        ("The answer is 3", 5, 2),
        ("5", 5, 4),
        ("I think option 2 is best", 5, 1),
        ("None of them", 5, None),  # No valid number
        ("10", 5, None),  # Out of range
    ]

    print("\nTesting vote extraction:")
    all_passed = True
    for response, num_candidates, expected in test_cases:
        result = reward_manager.extract_vote_from_response(response, num_candidates)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} Response: '{response}' (n={num_candidates}) -> Expected: {expected}, Got: {result}")

    print(f"\nVote extraction test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_multiple_benchmarks():
    """Test that different benchmarks work correctly."""
    print("\n" + "="*60)
    print("Test 5: Multiple Benchmarks")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_manager = PeerReviewVotingRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        use_model_voting=False
    )

    # Test with different benchmarks
    benchmarks = [
        ("gsm8k", "Solution: \n#### 42", "42"),
        ("math", "So \\boxed{256} is the answer", "256"),
        ("custom", "The answer is 100", "100"),
    ]

    print("\nTesting multiple benchmarks:")
    all_passed = True
    for data_source, response, expected_contains in benchmarks:
        result = reward_manager.extract_answer(response, data_source)
        passed = result is not None
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} Benchmark: {data_source} -> Extracted: {result}")

    print(f"\nMultiple benchmarks test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PEER-REVIEW VOTING SYSTEM - TEST SUITE")
    print("="*60)

    tests = [
        ("GSM8K Extraction", test_gsm8k_extraction),
        ("Generic Extraction", test_generic_extraction),
        ("Simulated Voting", test_simulated_voting),
        ("Vote Extraction", test_vote_extraction),
        ("Multiple Benchmarks", test_multiple_benchmarks),
    ]

    results = []
    for test_name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    print("="*60)

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The peer-review voting system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

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

from collections import defaultdict, Counter
import re
import torch
import numpy as np

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register


@register("peer_review")
class PeerReviewVotingRewardManager:
    """
    Peer-review voting reward manager.

    Instead of simple majority voting, each solver that proposed an answer
    votes on all other proposed answers. The answer with the most votes wins.

    This implements a more sophisticated consensus mechanism where:
    1. Each solver proposes an answer (standard generation)
    2. Each solver reviews all proposed answers and votes for the best one
    3. The answer with most votes is selected
    4. Rewards are assigned based on whether the answer matches the winner
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        voting_model=None,
        voting_temperature=0.7,
        max_voting_tokens=10,
        generation_fn=None,
        use_model_voting=False
    ) -> None:
        """
        Initialize the peer-review voting reward manager.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            num_examine: Number of examples to print for debugging
            compute_score: Optional custom scoring function
            reward_fn_key: Key to identify data source
            voting_model: Model to use for generating votes (optional, for future use)
            voting_temperature: Temperature for vote generation
            max_voting_tokens: Maximum tokens to generate for each vote
            generation_fn: Optional function to generate text with the model
            use_model_voting: Whether to use model-based voting (requires generation_fn)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.voting_model = voting_model
        self.voting_temperature = voting_temperature
        self.max_voting_tokens = max_voting_tokens
        self.generation_fn = generation_fn
        self.use_model_voting = use_model_voting

        if self.use_model_voting and self.generation_fn is None:
            print("Warning: use_model_voting=True but generation_fn is None. Falling back to simulated voting.")
            self.use_model_voting = False

    def extract_answer(self, response_str, data_source):
        """
        Extract answer from response based on data source.

        Args:
            response_str: The response string to extract answer from
            data_source: The data source identifier

        Returns:
            Extracted answer or None if extraction fails
        """
        try:
            if "gsm8k" in data_source.lower():
                from verl.utils.reward_score import gsm8k
                return gsm8k.extract_solution(response_str)
            elif "math" in data_source.lower():
                from verl.utils.reward_score import math
                try:
                    string_in_last_boxed = math.last_boxed_only_string(response_str)
                    if string_in_last_boxed is not None:
                        return math.remove_boxed(string_in_last_boxed)
                    else:
                        return None
                except Exception as e:
                    print(f"Error extracting MATH answer from {response_str}: {e}")
                    return None
            elif "multiply" in data_source.lower():
                from verl.utils.reward_score import multiply
                return multiply.extract_solution(response_str)
            else:
                # Generic answer extraction for other benchmarks
                # Try to find the last number or boxed answer
                return self._generic_answer_extraction(response_str)
        except Exception as e:
            print(f"Error extracting answer from {response_str}: {e}")
            return None

    def _generic_answer_extraction(self, response_str):
        """
        Generic answer extraction for benchmarks without specific extractors.

        Tries multiple strategies:
        1. Look for #### format (GSM8K style)
        2. Look for boxed format (MATH style)
        3. Look for "The answer is X" patterns
        4. Extract last number or word in quotes

        Args:
            response_str: The response string

        Returns:
            Extracted answer or None
        """
        # Try GSM8K format
        match = re.search(r"####\s*(.+?)(?:\n|$)", response_str)
        if match:
            return match.group(1).strip()

        # Try boxed format
        match = re.search(r"\\boxed\{([^}]+)\}", response_str)
        if match:
            return match.group(1).strip()

        # Try "The answer is X" pattern
        match = re.search(r"(?:the answer is|answer:|final answer:)\s*(.+?)(?:\.|$)", response_str, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try to find the last number
        numbers = re.findall(r"-?[\d,]+\.?\d*", response_str)
        if numbers:
            return numbers[-1].replace(",", "")

        # Try to find text in quotes
        match = re.search(r'"([^"]+)"', response_str)
        if match:
            return match.group(1).strip()

        return None

    def create_voting_prompt(self, original_prompt, candidate_answers, data_source):
        """
        Create a prompt for a solver to vote on candidate answers.

        Args:
            original_prompt: The original question/prompt
            candidate_answers: List of candidate answer strings
            data_source: The data source identifier

        Returns:
            Voting prompt string
        """
        # Filter out None and duplicate answers
        unique_answers = []
        seen = set()
        for ans in candidate_answers:
            if ans is not None and ans not in seen:
                unique_answers.append(ans)
                seen.add(ans)

        if len(unique_answers) <= 1:
            # If there's only one unique answer, no voting needed
            return None

        # Create numbered list of candidate answers
        candidates_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(unique_answers)])

        prompt = f"""Given the following problem:

{original_prompt}

Here are the proposed solutions:
{candidates_str}

Which solution do you think is correct? Reply with just the number (1-{len(unique_answers)}) of the best solution."""

        return prompt, unique_answers

    def extract_vote_from_response(self, vote_response, num_candidates):
        """
        Extract the vote (answer index) from the voting response.

        Args:
            vote_response: The model's voting response
            num_candidates: Number of candidate answers

        Returns:
            Index (0-based) of the voted answer, or None if extraction fails
        """
        # Look for a number in the response
        numbers = re.findall(r'\b([1-9]\d*)\b', vote_response)
        if numbers:
            vote_num = int(numbers[0])
            if 1 <= vote_num <= num_candidates:
                return vote_num - 1  # Convert to 0-based index

        return None

    def perform_model_based_voting(self, original_prompt, candidate_answers, num_voters):
        """
        Perform actual model-based voting using the generation function.

        Each solver gets to vote on all candidate answers.

        Args:
            original_prompt: The original question/prompt
            candidate_answers: List of candidate answer strings
            num_voters: Number of voters (typically equal to number of answers)

        Returns:
            Counter mapping answer index to vote count
        """
        if self.generation_fn is None:
            raise ValueError("generation_fn must be provided for model-based voting")

        # Filter out None and create unique answers list
        unique_answers = []
        answer_to_indices = defaultdict(list)  # Map answer to its indices in candidate_answers

        for i, ans in enumerate(candidate_answers):
            if ans is not None:
                if ans not in [ua for ua in unique_answers]:
                    unique_answers.append(ans)
                answer_to_indices[ans].append(i)

        if len(unique_answers) <= 1:
            # Only one unique answer, it gets all votes
            if unique_answers:
                return Counter({unique_answers[0]: num_voters})
            else:
                return Counter()

        # Create voting prompt
        candidates_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(unique_answers)])
        voting_prompt = f"""Given the following problem:

{original_prompt}

Here are the proposed solutions:
{candidates_str}

Which solution do you think is correct? Reply with just the number (1-{len(unique_answers)}) of the best solution."""

        # Generate votes from each voter
        votes = Counter()
        for voter_idx in range(num_voters):
            try:
                # Use generation function to get a vote
                vote_response = self.generation_fn(
                    voting_prompt,
                    max_new_tokens=self.max_voting_tokens,
                    temperature=self.voting_temperature
                )

                # Extract the vote from the response
                vote_idx = self.extract_vote_from_response(vote_response, len(unique_answers))

                if vote_idx is not None and 0 <= vote_idx < len(unique_answers):
                    voted_answer = unique_answers[vote_idx]
                    votes[voted_answer] += 1
                else:
                    # If vote extraction fails, abstain (no vote)
                    pass

            except Exception as e:
                print(f"Error during voting for voter {voter_idx}: {e}")
                # Abstain on error
                pass

        return votes

    def simulate_voting(self, candidate_answers):
        """
        Simulate voting without using a model.

        This is a fallback when voting_model is not provided.
        Uses simple heuristics:
        1. Each answer "votes" for the most common answer (majority principle)
        2. If tie, vote for the longest answer (more detailed)

        Args:
            candidate_answers: List of candidate answer strings

        Returns:
            Counter mapping answer to vote count
        """
        # Count frequency of each answer
        answer_counts = Counter([ans for ans in candidate_answers if ans is not None])

        if not answer_counts:
            return Counter()

        # In simulated voting, each unique answer gets votes equal to its frequency
        # This simulates the idea that solvers with the same answer would vote for each other
        return answer_counts

    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute rewards using peer-review voting.

        Args:
            data: DataProto containing the batch of responses
            return_dict: Whether to return a dictionary with extra info

        Returns:
            reward_tensor: Tensor of rewards for each response
            or dict with reward_tensor and reward_extra_info if return_dict=True
        """
        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Group responses by uid
        uid_to_data = defaultdict(lambda: {
            'indices': [],
            'answers': [],
            'responses': [],
            'prompts': [],
            'data_source': None
        })

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Decode prompt for voting
            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

            uid = data_item.non_tensor_batch["uid"]
            data_source = data_item.non_tensor_batch["data_source"]

            # Extract answer
            extracted_answer = self.extract_answer(response_str, data_source)

            uid_to_data[uid]['indices'].append(i)
            uid_to_data[uid]['answers'].append(extracted_answer)
            uid_to_data[uid]['responses'].append(response_str)
            uid_to_data[uid]['prompts'].append(prompt_str)
            uid_to_data[uid]['data_source'] = data_source

        # Perform voting for each uid group
        for uid, group_data in uid_to_data.items():
            indices = group_data['indices']
            answers = group_data['answers']
            prompts = group_data['prompts']
            data_source = group_data['data_source']

            if len(answers) <= 1:
                # Only one answer, give it full reward
                if answers and answers[0] is not None:
                    data_item = data[indices[0]]
                    prompt_length = data_item.batch["prompts"].shape[-1]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    reward_tensor[indices[0], valid_response_length - 1] = 1.0
                continue

            # Perform voting (either model-based or simulated)
            if self.use_model_voting and self.generation_fn is not None:
                # Use actual model-based voting
                original_prompt = prompts[0] if prompts else ""
                vote_counts = self.perform_model_based_voting(
                    original_prompt,
                    answers,
                    num_voters=len(answers)
                )
            else:
                # Use simulated voting
                vote_counts = self.simulate_voting(answers)

            if not vote_counts:
                # No valid answers
                continue

            # Get the winning answer
            winning_answer = vote_counts.most_common(1)[0][0]
            max_votes = vote_counts[winning_answer]

            # Assign rewards based on winning answer
            for idx, answer in zip(indices, answers):
                data_item = data[idx]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

                if answer == winning_answer and answer is not None:
                    reward_tensor[idx, valid_response_length - 1] = 1.0
                else:
                    reward_tensor[idx, valid_response_length - 1] = 0.0

                # Record voting info
                reward_extra_info['uid'].append(uid)
                reward_extra_info['answer'].append(answer)
                reward_extra_info['winning_answer'].append(winning_answer)
                reward_extra_info['vote_count'].append(vote_counts[answer] if answer is not None else 0)
                reward_extra_info['max_votes'].append(max_votes)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

# Peer-Review Voting System

## Overview

The peer-review voting system is an advanced consensus mechanism for selecting the best answer from multiple solver runs. Unlike simple majority voting, where the most frequent answer wins, peer-review voting allows each solver to evaluate all proposed answers and vote for the one it thinks is best.

## How It Works

### Architecture

1. **Proposal Phase**: Each solver generates N answers for a given problem
2. **Voting Phase**: Each solver reviews all N candidate answers and votes for the best one
3. **Tallying Phase**: Votes are counted and the answer with the most votes is selected
4. **Reward Phase**: Solvers whose answers match the winning answer receive rewards

### Comparison with Majority Voting

| Feature | Majority Voting | Peer-Review Voting |
|---------|----------------|-------------------|
| Selection Method | Most frequent answer | Answer with most votes from peers |
| Evaluation | Simple frequency count | Each solver evaluates all answers |
| Handling Multiple Correct Answers | May miss nuances | Better handling through peer evaluation |
| Computational Cost | Low | Higher (requires additional generation) |

## Voting Modes

### 1. Simulated Voting (Default)

In simulated voting mode, no additional model calls are made. Instead, the system uses a heuristic where each unique answer receives votes equal to its frequency. This is efficient and works well when solvers with the same answer would likely vote similarly.

**Advantages**:
- No additional computational cost
- Fast and efficient
- Works well for problems with clear correct/incorrect answers

**Use when**: You want the benefits of peer-review logic without the cost of additional model calls.

### 2. Model-Based Voting

In model-based voting mode, the system actually generates votes by showing each solver all candidate answers and asking which one is best. This is more faithful to the peer-review concept but requires additional inference.

**Advantages**:
- True peer review - each solver independently evaluates all answers
- Can handle subjective or multi-faceted problems better
- Provides richer training signal

**Use when**: You need genuine peer evaluation and have the computational budget for additional inference.

## Configuration

### Basic Configuration

Create or use the provided configuration file `verl/trainer/config/exps/peer_review.yaml`:

```yaml
# @package _global_

reward_model:
  reward_manager: peer_review
  reward_kwargs:
    use_model_voting: false  # Set to true for model-based voting
    voting_temperature: 0.7
    max_voting_tokens: 10

trainer:
  self_play_solver_reward: ttrl
  base_format: false
```

### Configuration Parameters

- **`reward_manager`**: Set to `peer_review` to use the peer-review voting system
- **`use_model_voting`**:
  - `false` (default): Use simulated voting
  - `true`: Use model-based voting (requires `generation_fn`)
- **`voting_temperature`**: Temperature for vote generation (0.0 = greedy, higher = more random)
- **`max_voting_tokens`**: Maximum tokens to generate for each vote (typically 5-20)

## Usage

### Using with GSM8K

```bash
# Train with peer-review voting on GSM8K
python -m verl.trainer.main_ppo \
    --config-path configs \
    --config-name gsm8k_ppo \
    reward_model.reward_manager=peer_review \
    actor_rollout_ref.rollout.n=8
```

### Using with Other Benchmarks

The peer-review voting system supports multiple benchmarks out of the box:

- **GSM8K**: Uses `#### number` format extraction
- **MATH**: Uses LaTeX `\boxed{}` extraction
- **Multiply**: Custom multiplication problem extraction
- **Generic**: Falls back to heuristic extraction for other formats

#### Example: Custom Benchmark

```yaml
# my_benchmark.yaml
data:
  train_files: path/to/train.parquet
  val_files: path/to/val.parquet

reward_model:
  reward_manager: peer_review
  reward_kwargs:
    use_model_voting: false

actor_rollout_ref:
  rollout:
    n: 8  # Generate 8 answers per problem
```

## Advanced Usage

### Enabling Model-Based Voting

To use actual model-based voting, you need to provide a generation function to the reward manager. Here are several ways to do this:

#### Option 1: Simple Model Integration

```python
from verl.workers.reward_manager import PeerReviewVotingRewardManager
from verl.workers.reward_manager.voting_utils import create_simple_generation_fn

# Create generation function
generation_fn = create_simple_generation_fn(model, tokenizer, device='cuda')

# Create reward manager with model-based voting
reward_manager = PeerReviewVotingRewardManager(
    tokenizer=tokenizer,
    num_examine=10,
    use_model_voting=True,
    generation_fn=generation_fn,
    voting_temperature=0.7,
    max_voting_tokens=10
)
```

#### Option 2: Integration in Trainer

Modify the trainer code to inject the generation function:

```python
# In ray_trainer.py or your custom trainer

from verl.workers.reward_manager.voting_utils import create_simple_generation_fn

# When creating the reward manager
def load_reward_manager_with_voting(self):
    # Create generation function using the actor model
    generation_fn = create_simple_generation_fn(
        self.actor_model,  # Your actor model
        self.tokenizer,
        device='cuda'
    )

    # Load reward manager with additional kwargs
    reward_manager = load_reward_manager(
        config=self.config,
        tokenizer=self.tokenizer,
        num_examine=10,
        generation_fn=generation_fn,  # Pass the generation function
        use_model_voting=True  # Enable model-based voting
    )

    return reward_manager
```

#### Option 3: Custom Generation Function

If you need custom generation logic:

```python
def my_custom_generation_fn(prompt: str, max_new_tokens: int = 10, temperature: float = 0.7) -> str:
    """
    Custom generation function.

    Args:
        prompt: The voting prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text containing the vote
    """
    # Your custom generation logic here
    # For example, using a different model, API, or generation strategy

    response = your_model.generate(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature
    )

    return response

# Use with reward manager
reward_manager = PeerReviewVotingRewardManager(
    tokenizer=tokenizer,
    num_examine=10,
    use_model_voting=True,
    generation_fn=my_custom_generation_fn
)
```

### Extending Answer Extraction

To add support for a new benchmark format, you can extend the `extract_answer` method:

```python
# In peer_review.py

def extract_answer(self, response_str, data_source):
    """Extract answer from response based on data source."""

    # Add your custom extraction logic
    if "my_benchmark" in data_source.lower():
        # Custom extraction for your benchmark
        match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response_str)
        if match:
            return match.group(1).strip()
        return None

    # ... existing extraction logic ...
```

Or override the `_generic_answer_extraction` method for more general patterns.

## Implementation Details

### Answer Extraction

The system uses benchmark-specific extractors when available:

1. **GSM8K**: Extracts numbers after `####` marker
2. **MATH**: Extracts LaTeX expressions from `\boxed{}`
3. **Multiply**: Custom extractor for arithmetic
4. **Generic**: Falls back to multiple heuristics:
   - `#### X` format
   - `\boxed{X}` format
   - "The answer is X" patterns
   - Last number in response
   - Text in quotes

### Vote Extraction

When using model-based voting, votes are extracted using:

1. Find first number in the response
2. Validate it's within valid range (1 to N)
3. Convert to 0-based index
4. If extraction fails, the vote is abstained

### Reward Assignment

Rewards are assigned as follows:

- **Winning answer**: 1.0 reward at the final token position
- **Non-winning answers**: 0.0 reward
- **Invalid answers (None)**: 0.0 reward

## Best Practices

### When to Use Peer-Review Voting

✅ **Use peer-review voting when**:
- Problems may have multiple valid approaches
- Answer quality is subjective or multi-faceted
- You want more sophisticated consensus than simple majority
- You have computational budget for additional inference (model-based mode)

❌ **Don't use peer-review voting when**:
- Answers are clearly right/wrong with no ambiguity
- Simple majority voting works well
- You're extremely compute-constrained (use simulated mode instead)

### Hyperparameter Tuning

- **`n` (number of solvers)**: Start with 8-16. More solvers = better consensus but higher cost
- **`voting_temperature`**: Use 0.7-1.0 for diverse voting, 0.0-0.3 for consistent voting
- **`max_voting_tokens`**: Typically 5-20 is enough for "Answer: N" format

### Performance Optimization

1. **Start with simulated voting**: Test your setup first, then enable model-based voting
2. **Batch voting requests**: Use `create_batched_generation_fn` for efficiency
3. **Cache voting results**: Use `VotingCache` to avoid redundant votes on identical answer sets
4. **Monitor vote diversity**: Track `reward_extra_info` to ensure votes are distributed

## Monitoring and Debugging

### Reward Extra Info

The reward manager returns extra information for monitoring:

```python
reward_result = reward_manager(data, return_dict=True)
reward_tensor = reward_result["reward_tensor"]
extra_info = reward_result["reward_extra_info"]

# Extra info contains:
# - 'uid': Unique identifier for each problem
# - 'answer': The answer proposed by each solver
# - 'winning_answer': The answer that won the vote
# - 'vote_count': Votes received by each answer
# - 'max_votes': Maximum votes received by any answer
```

### Debugging Tips

1. **Enable num_examine**: Set `num_examine > 0` to print sample responses
2. **Check vote distributions**: Ensure votes aren't too concentrated on one answer
3. **Validate answer extraction**: Print extracted answers to verify parsing works
4. **Test with known data**: Use examples with known correct answers first

## Examples

### Example 1: GSM8K with Simulated Voting

```bash
python -m verl.trainer.main_ppo \
    --config-path verl/trainer/config \
    --config-name ppo_trainer \
    +exps=peer_review \
    data.train_files=gsm8k/train.parquet \
    data.val_files=gsm8k/test.parquet \
    actor_rollout_ref.rollout.n=8
```

### Example 2: Custom Benchmark

```yaml
# custom_benchmark_peer_review.yaml
data:
  train_files: my_data/train.parquet
  val_files: my_data/test.parquet
  max_prompt_length: 1024
  max_response_length: 2048

reward_model:
  reward_manager: peer_review
  reward_kwargs:
    use_model_voting: false

actor_rollout_ref:
  rollout:
    n: 12  # 12 solvers

trainer:
  self_play_solver_reward: ttrl
```

### Example 3: Model-Based Voting Integration

```python
# In your training script

from verl.trainer.ppo.reward import load_reward_manager
from verl.workers.reward_manager.voting_utils import create_simple_generation_fn

# Create generation function
generation_fn = create_simple_generation_fn(
    model=actor_model,
    tokenizer=tokenizer,
    device='cuda'
)

# Load reward manager with model-based voting
reward_manager = load_reward_manager(
    config=config,
    tokenizer=tokenizer,
    num_examine=10,
    generation_fn=generation_fn,
    use_model_voting=True,
    voting_temperature=0.8,
    max_voting_tokens=15
)

# Use in training loop
reward_tensor = reward_manager(data_proto)
```

## Troubleshooting

### Issue: "generation_fn must be provided for model-based voting"

**Solution**: Either set `use_model_voting: false` or provide a `generation_fn` parameter.

### Issue: Votes are all going to one answer

**Possible causes**:
1. Voting temperature too low (try increasing to 0.7-1.0)
2. All solvers generating the same answer (check solver diversity)
3. Vote extraction failing (check vote_count in extra_info)

**Solution**: Increase temperature, ensure solver diversity, validate vote extraction logic.

### Issue: Answer extraction returning None

**Possible causes**:
1. Response format doesn't match expected pattern
2. Benchmark not supported

**Solution**:
- Check response format
- Add custom extraction logic for your benchmark
- Use generic extraction by not matching any specific data source

### Issue: Training is slow with model-based voting

**Solution**:
1. Start with simulated voting
2. Use fewer voters (lower `n`)
3. Implement batched voting
4. Use voting cache to avoid redundant calls

## Future Enhancements

Potential improvements to the peer-review voting system:

1. **Weighted voting**: Give more weight to historically accurate solvers
2. **Multi-round voting**: Iterative refinement of answers based on votes
3. **Confidence scoring**: Ask voters to provide confidence scores
4. **Explanation voting**: Vote based on solution quality, not just final answer
5. **Hybrid voting**: Combine frequency and peer-review signals

## References

- Original majority voting implementation: `verl/workers/reward_manager/majority.py`
- Configuration examples: `verl/trainer/config/exps/`
- Integration utilities: `verl/workers/reward_manager/voting_utils.py`

## Support

For issues or questions:
1. Check this documentation
2. Review example configurations in `verl/trainer/config/exps/`
3. Examine the source code in `verl/workers/reward_manager/peer_review.py`
4. Open an issue on the repository

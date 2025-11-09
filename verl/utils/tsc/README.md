# Tool-Grounded Self-Consistency (TSC) for VERL

## Overview

Tool-Grounded Self-Consistency (TSC) is a reward mechanism for reinforcement learning that combines:

1. **Tool-based verification**: External tools (unit tests, calculators, constraint checkers) evaluate solution correctness
2. **Self-consistency**: Agreement among multiple candidate solutions indicates confidence

The key insight is that **agreement only helps when tied to correctness**:
- Correct majorities are rewarded (high confidence + correct)
- Wrong majorities are penalized (high confidence but incorrect)
- Correct minorities are rewarded when majority is wrong (dissent from incorrect consensus)

This approach ensures the model learns to:
- Produce correct solutions verified by tools
- Build confidence through consistency when correct
- Avoid reinforcing confident but wrong answers

## Architecture

TSC is implemented as a modular extension to VERL's PPO/GRPO training pipeline:

```
Training Data (K candidates per problem)
    ↓
TSC Reward Manager
    ├─→ Tool Evaluation (tests, calculators, etc.)
    ├─→ Semantic Clustering (group equivalent solutions)
    └─→ TSC Reward Computation (tool + consistency)
    ↓
GRPO Advantage Estimation
    ↓
PPO Policy Update
```

## Components

### 1. Clustering (`clustering.py`)

Clusters candidate solutions by semantic equivalence:

- **Tool-based clustering**: Groups by test results (for code)
- **Numeric clustering**: Groups by extracted numeric value (for math)
- **Text-based clustering**: Groups by normalized text (for QA)

```python
from verl.utils.tsc import cluster_candidates

clusters = cluster_candidates(
    candidates=["4", "4.0", "5"],
    tool_results=[
        {'score': 1.0, 'passed': True},
        {'score': 1.0, 'passed': True},
        {'score': 0.0, 'passed': False},
    ],
    mode='auto',  # or 'tool', 'text', 'numeric'
)
```

### 2. Tool Evaluation (`tool_evaluation.py`)

Evaluates candidates using external tools:

- **MathToolEvaluator**: MATH, GSM8K datasets
- **CodeToolEvaluator**: CodeContests, APPS (with sandbox)
- **QAToolEvaluator**: SearchR1, TriviaQA, etc.
- **CustomToolEvaluator**: User-defined scoring functions

```python
from verl.utils.tsc import create_evaluator, evaluate_with_tools

evaluator = create_evaluator(
    tool_type='math',
    data_source='openai/gsm8k'
)

result = evaluate_with_tools(
    evaluator=evaluator,
    problem="What is 2+2?",
    candidate="The answer is 4",
    ground_truth="4",
)
# result.score = 1.0, result.passed = True
```

### 3. Reward Computation (`reward_computation.py`)

Computes TSC rewards based on tool scores and clusters:

**Reward Formula:**

For each candidate `i`:

1. **Base tool reward**: `r_tool = tool_score_i` (from tests/checks)
2. **Consistency component**: `r_consistency = cluster_fraction_i` (size of cluster / total candidates)

3. **Final reward** depends on cluster status:

   - **Correct majority**:
     ```
     reward = α * r_tool + β * r_consistency + bonus_correct_majority
     ```

   - **Wrong majority**:
     ```
     reward = α * r_tool - penalty_wrong_majority
     ```

   - **Correct minority** (when majority is wrong):
     ```
     reward = α * r_tool + bonus_correct_minority
     ```

```python
from verl.utils.tsc import compute_tsc_rewards, TSCConfig

config = TSCConfig(
    alpha_tool=0.8,          # Weight for tool correctness
    beta_consistency=0.2,     # Weight for self-consistency
    correct_majority_bonus=0.1,
    wrong_majority_penalty=0.3,
    correct_minority_bonus=0.15,
)

rewards = compute_tsc_rewards(
    candidates=candidates,
    tool_results=tool_results,
    clusters=clusters,
    config=config,
)
```

### 4. TSC Reward Manager (`reward_manager/tsc.py`)

Integrates TSC into VERL's training pipeline:

- Groups candidates by UID (problem ID)
- Evaluates all candidates with tools
- Clusters candidates
- Computes TSC rewards
- Logs detailed metrics

## Configuration

### YAML Config (`tsc_trainer.yaml`)

```yaml
critic:
  reward_manager: tsc

tsc:
  tool_type: auto              # auto, math, code, qa
  clustering_mode: auto         # auto, tool, text, numeric

  # Reward weights
  alpha_tool: 0.8
  beta_consistency: 0.2

  # Bonuses/penalties
  correct_majority_bonus: 0.1
  wrong_majority_penalty: 0.3
  correct_minority_bonus: 0.15

  # Thresholds
  min_cluster_frac_for_majority: 0.5
  tool_correct_threshold: 0.8

  # Options
  allow_correct_minority_reward: true
  normalize_rewards: true
  reward_scale: 1.0
  code_timeout: 10.0
  debug: false

algorithm:
  adv_estimator: grpo           # Use GRPO with TSC rewards
  norm_adv_by_std_in_grpo: true
```

### Python Config

```python
from verl.workers.reward_manager import TSCRewardManager

reward_manager = TSCRewardManager(
    tokenizer=tokenizer,
    num_examine=5,
    tool_type='math',
    alpha_tool=0.8,
    beta_consistency=0.2,
    # ... other params
)
```

## Data Preparation

TSC requires **K candidates per problem** in the training data. Prepare your dataset as follows:

### Parquet Schema

```python
{
    'prompt': str,              # Problem statement
    'uid': str,                 # Unique problem ID (same for K copies)
    'data_source': str,         # Dataset name (e.g., 'openai/gsm8k')
    'ground_truth': str,        # Answer or test cases (JSON)
    'extra_info': dict,         # Optional metadata
}
```

### Example: Creating TSC Dataset

```python
import pandas as pd

# Original dataset with unique problems
original_data = [
    {
        'prompt': 'What is 2+2?',
        'ground_truth': '4',
        'data_source': 'openai/gsm8k',
    },
    # ... more problems
]

# Replicate each problem K times with same UID
K = 4
tsc_data = []
for i, problem in enumerate(original_data):
    for _ in range(K):
        tsc_data.append({
            'prompt': problem['prompt'],
            'uid': f'problem_{i}',  # Same UID for K copies
            'data_source': problem['data_source'],
            'ground_truth': problem['ground_truth'],
        })

# Save to parquet
df = pd.DataFrame(tsc_data)
df.to_parquet('train_tsc.parquet')
```

**Note**: Each problem appears K times in the dataset with identical `uid` for grouping during reward computation.

## Usage Examples

### Example 1: Math (GSM8K)

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=tsc_trainer \
    data.train_files=~/data/gsm8k/train_tsc.parquet \
    data.val_files=~/data/gsm8k/test_tsc.parquet \
    actor_rollout_ref.model.path=deepseek-math-7b \
    algorithm.adv_estimator=grpo \
    critic.reward_manager=tsc \
    tsc.tool_type=math \
    tsc.clustering_mode=numeric
```

### Example 2: Code (CodeContests)

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=tsc_trainer \
    data.train_files=~/data/codecontests/train_tsc.parquet \
    actor_rollout_ref.model.path=deepseek-coder-7b \
    critic.reward_manager=tsc \
    tsc.tool_type=code \
    tsc.clustering_mode=tool \
    critic.sandbox_fusion.url=http://your-sandbox-url/run_code
```

## Metrics and Logging

TSC logs detailed metrics for monitoring:

### Per-Problem Metrics

- `tsc/avg_tool_score`: Average tool correctness
- `tsc/num_clusters`: Number of distinct solution clusters
- `tsc/cluster_entropy`: Diversity of solutions
- `tsc/has_majority`: Whether a majority cluster exists
- `tsc/majority_is_correct`: Whether majority is tool-correct
- `tsc/frac_correct_majority`: Fraction in correct majority
- `tsc/frac_wrong_majority`: Fraction in wrong majority
- `tsc/frac_correct_minority`: Fraction in correct minority

### Reward Metrics

- `tsc/avg_reward`: Mean reward across candidates
- `tsc/std_reward`: Reward variance
- `tsc/min_reward`, `tsc/max_reward`: Reward range

These metrics help diagnose:
- Are candidates diverse? (cluster_entropy)
- Is majority usually correct? (majority_is_correct)
- Are we rewarding wrong majorities? (frac_wrong_majority)

## Design Principles

### 1. No Learned Reward Models

TSC uses only:
- External tools (unit tests, calculators)
- Simple deterministic clustering
- Explicit reward formulas

**No** human labels or learned reward models are required.

### 2. Grounded in Verification

Rewards are **grounded** in verifiable correctness:
- Code passes tests
- Math matches ground truth
- QA satisfies constraints

Self-consistency alone is never rewarded without tool verification.

### 3. Robust to Confident Errors

The reward design explicitly handles:
- **Wrong majority**: Penalized (prevents reinforcing plausible but incorrect answers)
- **Correct minority**: Rewarded (encourages dissent from wrong consensus)

This prevents the model from learning "polite but wrong" behavior.

### 4. Configurable and Modular

All components are:
- Configurable via YAML
- Extensible (add new tool types, clustering modes)
- Composable with existing VERL infrastructure

## Best Practices

### 1. Choose K Based on Budget

- **K=4**: Good balance for most tasks
- **K=8-16**: Better signal, higher cost
- **K=2**: Minimum for TSC, limited self-consistency signal

### 2. Tune Weights for Your Task

**Math/QA** (deterministic answers):
- Higher `alpha_tool` (0.8-0.9)
- Lower `beta_consistency` (0.1-0.2)

**Code** (multiple valid solutions):
- Balanced weights (0.7-0.8 tool, 0.2-0.3 consistency)

### 3. Adjust Penalties/Bonuses

- **High-stakes correctness**: Increase `wrong_majority_penalty`
- **Encourage exploration**: Increase `correct_minority_bonus`

### 4. Monitor Metrics

Watch for:
- **Low cluster_entropy**: Candidates too similar, increase sampling temperature
- **High frac_wrong_majority**: Wrong answers dominating, tune weights
- **Low majority_is_correct**: Tool evaluator may be misconfigured

## Troubleshooting

### Issue: All candidates identical

**Solution**: Increase sampling diversity
- Raise `rollout.temperature` (0.7 → 1.0)
- Raise `rollout.top_p` (0.9 → 0.95)

### Issue: TSC rewards always zero

**Diagnosis**: Check tool evaluation
- Verify `data_source` matches evaluator
- Check `ground_truth` format
- Enable `tsc.debug=true` for detailed logging

### Issue: Wrong majorities not penalized

**Solution**: Adjust thresholds
- Lower `min_cluster_frac_for_majority` (0.5 → 0.4)
- Increase `wrong_majority_penalty` (0.3 → 0.5)

## Testing

Run tests to verify installation:

```bash
pytest tests/utils/tsc/ -v
```

Tests cover:
- Clustering correctness
- Reward computation logic
- Tool evaluation integration
- Edge cases (empty inputs, single candidates, etc.)

## Citation

If you use TSC in your research, please cite:

```bibtex
@software{verl_tsc_2025,
  title={Tool-Grounded Self-Consistency for VERL},
  author={VERL Team},
  year={2025},
  url={https://github.com/volcengine/verl}
}
```

## License

TSC is part of VERL and follows the same Apache 2.0 license.

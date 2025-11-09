#!/bin/bash
# Example training script for Tool-Grounded Self-Consistency (TSC)
# on math problems (GSM8K)

set -x

# Number of candidates per problem (K in TSC)
# The training data should contain K copies of each problem with the same UID
NUM_CANDIDATES=4

# Launch TSC training with GRPO
python3 -m verl.trainer.main_ppo \
    --config-name=tsc_trainer \
    data.train_files=~/data/rlhf/gsm8k/train_tsc.parquet \
    data.val_files=~/data/rlhf/gsm8k/test_tsc.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=~/models/deepseek-math-7b-base \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.4 \
    rollout.tensor_model_parallel_size=1 \
    rollout.log_prob_micro_batch_size=32 \
    rollout.temperature=0.8 \
    rollout.top_p=0.95 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    critic.reward_manager=tsc \
    tsc.tool_type=math \
    tsc.clustering_mode=numeric \
    tsc.alpha_tool=0.8 \
    tsc.beta_consistency=0.2 \
    tsc.correct_majority_bonus=0.1 \
    tsc.wrong_majority_penalty=0.3 \
    tsc.correct_minority_bonus=0.15 \
    tsc.min_cluster_frac_for_majority=0.5 \
    tsc.tool_correct_threshold=0.8 \
    tsc.allow_correct_minority_reward=true \
    tsc.normalize_rewards=true \
    tsc.reward_scale=1.0 \
    tsc.debug=false \
    trainer.total_epochs=3 \
    trainer.project_name=tsc_gsm8k \
    trainer.experiment_name=tsc_k${NUM_CANDIDATES}_deepseek7b \
    trainer.logger=['console','tracking'] \
    trainer.default_tracking=tensorboard

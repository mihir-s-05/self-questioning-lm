#!/bin/bash
# Example training script for Tool-Grounded Self-Consistency (TSC)
# on code problems (CodeContests)

set -x

# Number of candidates per problem (K in TSC)
NUM_CANDIDATES=8

# Sandbox URL (optional - if not set, will use local execution)
SANDBOX_URL="${SANDBOX_FUSION_URL:-null}"

# Launch TSC training with GRPO
python3 -m verl.trainer.main_ppo \
    --config-name=tsc_trainer \
    data.train_files=~/data/rlhf/codecontests/train_tsc.parquet \
    data.val_files=~/data/rlhf/codecontests/test_tsc.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=~/models/deepseek-coder-7b-instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    rollout.name=vllm \
    rollout.gpu_memory_utilization=0.4 \
    rollout.tensor_model_parallel_size=1 \
    rollout.log_prob_micro_batch_size=16 \
    rollout.temperature=0.7 \
    rollout.top_p=0.95 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    critic.reward_manager=tsc \
    critic.sandbox_fusion.url=${SANDBOX_URL} \
    tsc.tool_type=code \
    tsc.clustering_mode=tool \
    tsc.alpha_tool=0.85 \
    tsc.beta_consistency=0.15 \
    tsc.correct_majority_bonus=0.15 \
    tsc.wrong_majority_penalty=0.4 \
    tsc.correct_minority_bonus=0.2 \
    tsc.min_cluster_frac_for_majority=0.5 \
    tsc.tool_correct_threshold=0.8 \
    tsc.allow_correct_minority_reward=true \
    tsc.normalize_rewards=true \
    tsc.reward_scale=1.0 \
    tsc.code_timeout=10.0 \
    tsc.debug=false \
    trainer.total_epochs=5 \
    trainer.project_name=tsc_codecontests \
    trainer.experiment_name=tsc_k${NUM_CANDIDATES}_deepseek_coder7b \
    trainer.logger=['console','tracking'] \
    trainer.default_tracking=tensorboard

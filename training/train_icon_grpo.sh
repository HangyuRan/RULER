#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${VLM_ENDPOINTS:?Set VLM_ENDPOINTS to one or more comma-separated OpenAI-compatible VLM base URLs}"
: "${HF_MODEL_PATH:?Set HF_MODEL_PATH to a model directory or Hugging Face model ID}"
: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to the training parquet file}"
export VLM_ENDPOINTS HF_MODEL_PATH TRAIN_DATA_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export WANDB_MODE="${WANDB_MODE:-offline}"
ENGINE=vllm
ROLLOUT_NUMBER=8
GEN_TP=8

export VLM_MODEL="${VLM_MODEL:-qwen3-vl-8b-judge}"
export VLM_TIMEOUT="${VLM_TIMEOUT:-180}"
export PER_ENDPOINT_CONCURRENCY="${PER_ENDPOINT_CONCURRENCY:-32}"
export BLANK_MSE_THRESHOLD="${BLANK_MSE_THRESHOLD:-0.0001}"

export REWARD_SCRIPT_PATH="${REWARD_SCRIPT_PATH:-${SCRIPT_DIR}/reward/verl_reward.py}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/training/icon}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/icon}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
export DEBUG_BLANK_DIR="${DEBUG_BLANK_DIR:-${OUTPUT_DIR}/blank_images}"
export REWARD_LOG_PATH="${REWARD_LOG_PATH:-${OUTPUT_DIR}/reward_scores.jsonl}"
export DEBUG_PROMPT_LOG_PATH="${DEBUG_PROMPT_LOG_PATH:-${OUTPUT_DIR}/debug_api_payloads.jsonl}"
export PYTHONPATH="${SCRIPT_DIR}/reward${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

python -m verl.trainer.main_ppo --config-path=config \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_DATA_PATH" \
  data.val_files="$TRAIN_DATA_PATH" \
  data.train_batch_size=128 \
  ++data.apply_chat_template_kwargs.enable_thinking=False \
  data.max_prompt_length=512 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.shuffle=True \
  data.truncation='error' \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
  actor_rollout_ref.model.path="$HF_MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
  actor_rollout_ref.rollout.n=$ROLLOUT_NUMBER \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.optimizer_offload=False \
  reward_model.reward_manager=naive \
  custom_reward_function.path="$REWARD_SCRIPT_PATH" \
  custom_reward_function.name=compute_score_fast \
  trainer.logger='["console","tensorboard","wandb"]' \
  trainer.project_name="${PROJECT_NAME:-ruler_icon_grpo}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-icon_grpo}" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=1000 \
  trainer.val_before_train=False \
  trainer.total_epochs=4 \
  "$@"

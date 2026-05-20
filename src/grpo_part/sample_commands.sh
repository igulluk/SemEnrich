#!/bin/bash
# Sample Command Lines for GRPO Training Pipeline
# ================================================
#
# NOTE: The sft.py, grpo.py, and grpo_trainer.py files in this folder are 
# updated versions based on the open-r1-multimodal repository:
# https://github.com/EvolvingLMMs-Lab/open-r1-multimodal
#
# Key modifications include custom reward functions for medical report generation,
# integration with clustering-based semantic similarity rewards, and support for
# domain-specific evaluation metrics.
# ================================================


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

export DEBUG_MODE="false"
export LOG_PATH="./debug_log.txt"


# =============================================================================
# 1. SFT (Supervised Fine-Tuning) - sft.py
# =============================================================================

# SFT with Qwen2.5-VL-3B-Instruct
torchrun --nproc_per_node="4" \
    src/grpo_part/sft.py \
    --output_dir ./checkpoints/Qwen2.5-VL-3B/SFT \
    --data_json_path ./data/train_data.json \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name RexGradient \
    --deepspeed ./configs/zero3.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --num_train_epochs 200 \
    --run_name Qwen2.5-VL-3B-RexGrad-SFT \
    --save_only_model false \
    --eval_strategy steps \
    --save_steps 50 \
    --eval_steps 50 \
    --seed 42 \
    --learning_rate 1e-4 \
    --dataloader_num_workers 8 \
    --save_total_limit 10 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --early_stopping_patience 20



# =============================================================================
# 2. GRPO (Group Relative Policy Optimization) - grpo.py
# =============================================================================

# GRPO with accuracy and format rewards
torchrun --nproc_per_node="2" \
    src/grpo_part/grpo.py \
    --output_dir ./checkpoints/Qwen2.5-VL-3B/GRPO/acc_format \
    --reward_funcs accuracy format \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --sft_checkpoint_path ./checkpoints/Qwen2.5-VL-3B/SFT/checkpoint-best \
    --data_json_path ./data/train_data.json \
    --dataset_name RexGradient \
    --deepspeed ./configs/zero2.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --num_train_epochs 400 \
    --run_name Qwen2.5-VL-3B-RexGrad-GRPO-acc-format \
    --save_steps 50 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_only_model false \
    --seed 42 \
    --dataloader_num_workers 8 \
    --save_total_limit 10 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_reward \
    --greater_is_better true \
    --early_stopping_patience 25 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant \
    --num_generations 8

# GRPO with accuracy, format, and semantic cluster rewards (HDBSCAN)
torchrun --nproc_per_node="2" \
    src/grpo_part/grpo.py \
    --output_dir ./checkpoints/Qwen2.5-VL-3B/GRPO/acc_format_semantic \
    --reward_funcs accuracy format \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --sft_checkpoint_path ./checkpoints/Qwen2.5-VL-3B/SFT/checkpoint-best \
    --data_json_path ./data/train_data.json \
    --dataset_name RexGradient \
    --deepspeed ./configs/zero2.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --num_train_epochs 400 \
    --run_name Qwen2.5-VL-3B-RexGrad-GRPO-semantic \
    --save_steps 50 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_only_model false \
    --findings_cluster_path ./experiments/outputs/clustering/hdbscan_clusters.json \
    --impression_cluster_path ./experiments/outputs/clustering_impression/hdbscan_clusters.json \
    --seed 42 \
    --dataloader_num_workers 8 \
    --save_total_limit 10 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_reward \
    --greater_is_better true \
    --early_stopping_patience 25 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant \
    --num_generations 8


# =============================================================================
# 3. INFERENCE - inference.py
# =============================================================================


# Inference with Qwen2-VL-2B model
python src/grpo_part/inference.py \
    --checkpoint_path ./checkpoints/Qwen2-VL-2B/GRPO/acc_format_semantic/checkpoint-best \
    --sft_checkpoint_path ./checkpoints/Qwen2-VL-2B/SFT/checkpoint-best \
    --base_model_name Qwen/Qwen2-VL-2B-Instruct \
    --data_path ./data/test_data.json \
    --output_path ./outputs/Qwen2-VL-2B_inference_results.json \
    --max_new_tokens 512 \
    --seed 42 \
    --num_samples 500


# Batch inference with multiple seeds for statistical analysis
for seed in 42 123 456 789 1024; do
    python src/grpo_part/inference.py \
        --checkpoint_path ./checkpoints/Qwen2.5-VL-3B/GRPO/acc_format_semantic/checkpoint-best \
        --sft_checkpoint_path ./checkpoints/Qwen2.5-VL-3B/SFT/checkpoint-best \
        --base_model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --data_path ./data/test_data.json \
        --output_path ./outputs/inference_results_seed_${seed}.json \
        --max_new_tokens 512 \
        --seed $seed \
        --num_samples 500
done

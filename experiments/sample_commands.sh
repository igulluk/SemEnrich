#!/bin/bash
# Sample Command Lines for RexGrad Pipeline
# =============================================

# =============================================================================
# 1. CLUSTERING (run_clustering.py)
# =============================================================================

# K-Means clustering
python experiments/scripts/run_clustering.py \
    --config experiments/configs/clustering_kmeans/config_K2000.yaml

# HDBSCAN clustering
python experiments/scripts/run_clustering.py \
    --config experiments/configs/clustering_hdbscan/config.yaml

# DBSCAN clustering
python experiments/scripts/run_clustering.py \
    --config experiments/configs/clustering_dbscan/config.yaml


# =============================================================================
# 2. SIGN DETERMINATION (run_sign_determination.py)
# =============================================================================

# Sign determination for K-Means clusters
python experiments/scripts/run_sign_determination.py \
    --config experiments/configs/sign_determination/config_kmeans_K2000.yaml

# Sign determination for HDBSCAN clusters
python experiments/scripts/run_sign_determination.py \
    --config experiments/configs/sign_determination/config_hdbscan.yaml

# Sign determination for DBSCAN clusters
python experiments/scripts/run_sign_determination.py \
    --config experiments/configs/sign_determination/config_dbscan.yaml


# =============================================================================
# 3. EXPANSION (run_expansion.py)
# =============================================================================

# Expansion for K-Means
python experiments/scripts/run_expansion.py \
    --config experiments/configs/expansion/config_kmeans_K2000.yaml

# Expansion for HDBSCAN
python experiments/scripts/run_expansion.py \
    --config experiments/configs/expansion/config_hdbscan.yaml

# Expansion for DBSCAN
python experiments/scripts/run_expansion.py \
    --config experiments/configs/expansion/config_dbscan.yaml


# =============================================================================
# 4. TRAINING (run_training.py)
# =============================================================================

# Training with DeepSeek-R1-Distill-Qwen-1.5B model (K-Means clustering)
python experiments/scripts/run_training.py \
    --config experiments/configs/training/training.yaml \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --experiment_name DS-R1-Qwen-1.5B-kmeans-expansion \
    --dataset_list rexgrad-only-findings-nsamples-cluster_method-kmeans-ncluster-2K-expansion \
    --max_epochs 18 \
    --seq_length 512 \
    --batch_size 18 \
    --accumulation 8 \
    --project RexGradModel

# Training with DeepSeek-R1-Distill-Llama-8B model (HDBSCAN clustering)
python experiments/scripts/run_training.py \
    --config experiments/configs/training/training.yaml \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --experiment_name DS-R1-Llama-8B-hdbscan-expansion \
    --dataset_list rexgrad-only-findings-nsamples-cluster_method-hdbscan-expansion \
    --max_epochs 18 \
    --seq_length 512 \
    --batch_size 18 \
    --accumulation 8 \
    --project RexGradModel

# Training with Gemma3-4B model
python experiments/scripts/run_training.py \
    --config experiments/configs/training/training.yaml \
    --model_name google/gemma-3-4b \
    --experiment_name Gemma3-4B-dbscan-expansion \
    --dataset_list rexgrad-only-findings-nsamples-cluster_method-dbscan-expansion \
    --max_epochs 18 \
    --batch_size 18 \
    --accumulation 8 \
    --project RexGradModel


# =============================================================================
# 5. INFERENCE (run_inference.py)
# =============================================================================

# Inference with DeepSeek-R1-Distill-Qwen-1.5B model
python experiments/scripts/run_inference.py \
    --config experiments/configs/training/training.yaml \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --experiment_name DS-R1-Qwen-1.5B-kmeans-expansion-Inference \
    --dataset_list rexgrad-only-findings-nsamples \
    --ckpt_path /path/to/checkpoints/DS-R1-Qwen-1.5B-kmeans-expansion/best.ckpt \
    --seq_length 512 \
    --batch_size 18 \
    --project RexGradModel

# Inference with DeepSeek-R1-Distill-Llama-8B model
python experiments/scripts/run_inference.py \
    --config experiments/configs/training/training.yaml \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --experiment_name DS-R1-Llama-8B-hdbscan-expansion-Inference \
    --dataset_list rexgrad-only-findings-nsamples \
    --ckpt_path /path/to/checkpoints/DS-R1-Llama-8B-hdbscan-expansion/best.ckpt \
    --seq_length 512 \
    --batch_size 18 \
    --project RexGradModel

# Inference with Mistral-7B model
python experiments/scripts/run_inference.py \
    --config experiments/configs/training/training.yaml \
    --model_name mistralai/Mistral-7B-v0.1 \
    --experiment_name Mistral-7B-kmeans-expansion-Inference \
    --dataset_list rexgrad-only-findings-nsamples \
    --ckpt_path /path/to/checkpoints/Mistral-7B-kmeans-expansion/best.ckpt \
    --seq_length 512 \
    --project RexGradModel

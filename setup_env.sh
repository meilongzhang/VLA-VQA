#!/bin/bash

# FRAMES-VQA Environment Configuration
# This file contains environment variables to replace personal paths throughout the repository
# Source this file before running any scripts: source setup_env.sh

# ====== HOME ROOT ======
export HOME="/coc/testnvme/chuang475"  # Adjust to your home directory

# ====== PROJECT PATHS ======
export FRAMES_VQA_ROOT="${PWD}"

# ====== CONDA/PYTHON ENVIRONMENT ======
export CONDA_BASE_PATH="${HOME}/miniconda3"  # Adjust to your conda installation
# export CONDA_ENV_NAME="frames-vqa"  # Adjust to your environment name
export CONDA_ENV_NAME="lavis_same"  # Adjust to your environment name
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"

# ====== DATA PATHS ======
export DATASET_ROOT="${HOME}/datasets"  # Adjust to your dataset location
export NLTK_DATA_PATH="${DATASET_ROOT}/nltk_data"  # NLTK data directory

# ====== MODEL CHECKPOINTS ======
export CHECKPOINT_ROOT="${HOME}/checkpoints"  # Base directory for model checkpoints
export LLAVA_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/llava"
export PALIGEMMA_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/paligemma"

# ====== OUTPUT DIRECTORIES ======
export OUTPUT_ROOT="${FRAMES_VQA_ROOT}/output"
export LLAVA_OUTPUT_ROOT="${OUTPUT_ROOT}/LLAVA"
export PALIGEMMA_OUTPUT_ROOT="${OUTPUT_ROOT}/PALIGEMMA"

# ====== MODEL CONFIG PATHS ======
export LLAVA_CONFIG_ROOT="${FRAMES_VQA_ROOT}/configs/models/llava_vqa"
export PALIGEMMA_CONFIG_ROOT="${FRAMES_VQA_ROOT}/configs/models/paligemma_vqa"

# ====== CUDA SETTINGS FOR REPRODUCIBILITY ======
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

# Create necessary directories
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LLAVA_OUTPUT_ROOT}"
mkdir -p "${PALIGEMMA_OUTPUT_ROOT}"
mkdir -p "${CHECKPOINT_ROOT}"
mkdir -p "${LLAVA_CHECKPOINT_ROOT}"
mkdir -p "${PALIGEMMA_CHECKPOINT_ROOT}"
mkdir -p "${DATASET_ROOT}"
mkdir -p "${NLTK_DATA_PATH}"

echo "Environment variables set for FRAMES-VQA"
echo "FRAMES_VQA_ROOT: ${FRAMES_VQA_ROOT}"
echo "CONDA_ENV_NAME: ${CONDA_ENV_NAME}"
echo "DATASET_ROOT: ${DATASET_ROOT}"
echo "CHECKPOINT_ROOT: ${CHECKPOINT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo ""
echo "Please adjust the paths in this file according to your setup before using!"

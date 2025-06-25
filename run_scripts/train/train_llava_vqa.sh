#!/bin/bash

#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G

# Source environment variables
source setup_env.sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

srun -u ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path projects/llava/vqav2_train.yaml --options run.init_lr=$init_lr run.min_lr=$min_lr run.warmup_lr=$warmup_lr run.weight_decay=$weight_decay run.opt=$opt run.adamp_k=$adamp_k run.output_dir=$output_dir model.linear_probe=$linear_probe model.use_lora=$use_lora
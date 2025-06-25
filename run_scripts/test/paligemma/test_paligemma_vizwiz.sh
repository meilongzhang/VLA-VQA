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

srun -u ${PYTHON_BIN} -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path projects/paligemma/vizwiz_test.yaml --options model.wise=$wise model.finetuned=$finetuned model.use_lora=$use_lora
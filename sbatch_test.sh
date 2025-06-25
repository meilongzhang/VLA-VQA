#!/bin/bash

wise=0
use_lora=1
finetuned='output/PALIGEMMA/VQA/ft/checkpoint_best.pth'
# finetuned='output/PALIGEMMA/GQA/ft/checkpoint_best.pth'

name="val_paligemma_gqa"
job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/PALIGEMMA/eval/${job_name}"
mkdir -p "$output_dir"
sbatch --export "ALL,wise=${wise},finetuned=${finetuned},use_lora=${use_lora}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/test/paligemma/${name}.sh


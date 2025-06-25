#!/bin/bash

name="train_paligemma_gqa"
# name="train_paligemma_vqa"
# name="train_llava_vqa"

init_lr=1e-4
min_lr=1e-5
warmup_lr=1e-5
weight_decay=1e-5
linear_probe=0
use_lora=1
opt="adam"
adamp_k=0

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/PALIGEMMA/train/${job_name}"

mkdir -p "$output_dir"
sbatch --export "ALL,init_lr=${init_lr},min_lr=${min_lr},warmup_lr=${warmup_lr},weight_decay=${weight_decay},opt=${opt},adamp_k=${adamp_k},output_dir=${output_dir},linear_probe=${linear_probe},use_lora=${use_lora}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/train/${name}.sh
#!/bin/bash

wise=0
use_lora=0
finetuned='None'
# finetuned='/coc/testnvme/chuang475/projects/FRAMES-VQA/output/PALIGEMMA/VQA/ft/checkpoint_best.pth'
# finetuned='output/PALIGEMMA/GQA/ft/checkpoint_best.pth'

model='chatvla'
dataset='textvqa'

name="test_${model}_${dataset}"
job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/${model}/eval/${job_name}"
mkdir -p "$output_dir"
sbatch --export "ALL,wise=${wise},finetuned=${finetuned},use_lora=${use_lora}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/test/${model}/${name}.sh


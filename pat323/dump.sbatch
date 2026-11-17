#!/bin/bash
#SBATCH --job-name=p323d
#SBATCH --output=pat323/logs/%j_dump.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
# PAT-323: rolling-origin residual dump (inference only). usage: sbatch pat323/dump.sbatch OUT.pkl [max_systems]
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export PYTHONWARNINGS=ignore
.venv/bin/python pat323/dump_resid.py "$@"

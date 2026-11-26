#!/bin/bash
#SBATCH --job-name=p342_rot
#SBATCH --output=pat342/logs/%j_rot.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm54,elm55
# PAT-342: cumulative rotation of the transport (pat342/rot_cum.py). usage: sbatch pat342/rot.sbatch OUT CKPT [CKPT ...]
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export P336_LIE_RANK=4
.venv/bin/python -u -m pat342.rot_cum "$@" 2>&1 | grep -vE "Some weights|IS expected|IS NOT|TRAIN this"
echo done

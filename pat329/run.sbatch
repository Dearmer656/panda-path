#!/bin/bash
#SBATCH --job-name=p329_smap
#SBATCH --output=pat329/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# PAT-329: classical training-free S-map / simplex on own history vs S-map on FreshCast weights
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}
for H in 128 256 512; do $PY pat329/smap_classic.py --H $H > ${L}_H${H}.txt 2>&1 & done; wait
echo done

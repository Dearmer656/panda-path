#!/bin/bash
#SBATCH --job-name=p328_mat
#SBATCH --output=pat328/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# PAT-328: user retrieval transform (pair-m / path-m matrices on retrieved futures), frozen fc retrieval
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat328/logs/${SLURM_JOB_ID}
for H in 128 256 512; do for arm in pairm pathm; do $PY pat328/fc_mat.py --stage train --arm $arm --H $H > ${L}_train_${arm}_H${H}.txt 2>&1 & done; done; wait
$PY pat328/fc_mat.py --stage test > ${L}_test.txt 2>&1
echo done

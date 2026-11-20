#!/bin/bash
#SBATCH --job-name=p330_trunc
#SBATCH --output=pat330/logs/%j_trunc.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
# PAT-330 truncated-backbone baseline (backbone sees the last 512 steps) at the ctx 1024 / 2048 test origins
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat330/logs/${SLURM_JOB_ID}
for C in 1024 2048; do P321_LQ=96 P321_CTX=$C P321_OUT=pat321/data_ctx${C}_path $PY pat330/trunc.py > ${L}_trunc_ctx$C.txt 2>&1 & done; wait
echo done

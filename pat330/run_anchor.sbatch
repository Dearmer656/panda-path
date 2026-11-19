#!/bin/bash
#SBATCH --job-name=p330_anc
#SBATCH --output=pat330/logs/%j_anc.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# PAT-330 step 2: multi-anchor (each 16-step segment as anchor, later segments validate), one process per H
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat330/logs/${SLURM_JOB_ID}; export P321_LQ=96 P321_OUT=pat321/data
for H in 128 256 512; do $PY pat330/anchor.py --H $H > ${L}_anchor_H${H}.txt 2>&1 & done; wait
echo done

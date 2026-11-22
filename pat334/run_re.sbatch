#!/bin/bash
#SBATCH --job-name=p334_re
#SBATCH --output=pat334/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
# PAT-334 B re-anchored retrieval; args = full|ctx H
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
if [ $1 = full ]; then ENV="P321_OUT=pat321/data P333_HIST=full"; else ENV="P321_CTX=2048 P321_OUT=pat321/data_ctx2048_path P333_HIST=ctx"; fi
env OMP_NUM_THREADS=8 P321_LQ=96 $ENV P334_H=$2 .venv/bin/python pat334/reanchor.py > pat334/logs/${SLURM_JOB_ID}_re_$1_H$2.txt 2>&1

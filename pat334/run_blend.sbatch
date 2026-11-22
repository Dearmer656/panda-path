#!/bin/bash
#SBATCH --job-name=p334_blend
#SBATCH --output=pat334/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
# PAT-334 backbone + retrieval blend; arg = full|ctx, three H in parallel
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python
if [ $1 = full ]; then ENV="P321_OUT=pat321/data P333_HIST=full"; else ENV="P321_CTX=2048 P321_OUT=pat321/data_ctx2048_path P333_HIST=ctx"; fi
for H in 128 256 512; do env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 P321_LQ=96 $ENV P334_H=$H $PY pat334/blend.py > pat334/logs/${SLURM_JOB_ID}_blend_$1_H$H.txt 2>&1 & done
wait
echo done

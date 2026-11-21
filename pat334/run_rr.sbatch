#!/bin/bash
#SBATCH --job-name=p334_rr
#SBATCH --output=pat334/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
# PAT-334 (a) rerank sharded: two shards per GPU, args = shard specs 'set:i:n'
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python
for s in "$@"; do OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 P334_SHARD=$s P321_LQ=96 P321_OUT=pat321/data $PY pat334/rr.py > pat334/logs/${SLURM_JOB_ID}_${s//:/_}.txt 2>&1 & done
wait
echo done

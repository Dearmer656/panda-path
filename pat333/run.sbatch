#!/bin/bash
#SBATCH --job-name=p333_sig
#SBATCH --output=pat333/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
# PAT-333 signal-space retrieval batch: E1 thesis / E2 joint / E3 gate (ctx512 full, ctx2048 fair) + E4 context augmentation
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python
P321_LQ=96 P321_OUT=pat321/data P333_HIST=full $PY pat333/sig.py > pat333/logs/${SLURM_JOB_ID}_full512.txt 2>&1 &
P321_LQ=96 P321_CTX=2048 P321_OUT=pat321/data_ctx2048_path P333_HIST=ctx $PY pat333/sig.py > pat333/logs/${SLURM_JOB_ID}_fair2048.txt 2>&1 &
P321_LQ=96 P321_OUT=pat321/data $PY pat333/ctxaug.py > pat333/logs/${SLURM_JOB_ID}_aug.txt 2>&1 &
wait
echo done

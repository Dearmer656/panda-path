#!/bin/bash
#SBATCH --job-name=p329_msectx
#SBATCH --output=pat329/logs/%j_msectx.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=16:00:00
# PAT-329: MSE at ctx 1024 / 2048 (wave 1), S-map coordinate stride at ctx 512 (wave 2), one GPU
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}
for c in 1024 2048; do
  P321_LQ=96 P321_CTX=$c P321_OUT=pat321/data_ctx${c}_path $PY pat329/mse_eval.py > ${L}_mse_ctx${c}.txt 2>&1 &
  P321_LQ=96 P321_CTX=$c P321_OUT=pat321/data_ctx${c}_rope P329_ONLY=panda $PY pat329/mse_eval.py > ${L}_mse_ctx${c}_rope.txt 2>&1 &
done; wait
P321_LQ=96 P321_OUT=pat321/data $PY pat329/stride.py > ${L}_stride96.txt 2>&1 &
P321_LQ=64 P321_OUT=pat329/lq64 $PY pat329/stride.py > ${L}_stride64.txt 2>&1 &
wait
echo done

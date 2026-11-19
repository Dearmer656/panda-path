#!/bin/bash
#SBATCH --job-name=p329_topk
#SBATCH --output=pat329/logs/%j_topk.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=16:00:00
# PAT-329 top-k neighbours on FreshCast weights: all LQ (32/64/96) x H (128/256/512) packed on one GPU
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}_topk
for q in 32 64 96; do
  if [ $q = 96 ]; then O=pat321/data; else O=pat329/lq$q; fi
  for H in 128 256 512; do P321_LQ=$q P321_OUT=$O $PY pat329/topk.py --H $H --bs 64 > ${L}${q}_H${H}.txt 2>&1 & done
done; wait
echo done

#!/bin/bash
#SBATCH --job-name=p329_topk
#SBATCH --output=pat329/logs/%j_topk.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# PAT-329 top-k neighbours on FreshCast weights, LQ=$1
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}_topk$1
export P321_LQ=$1; if [ "$1" = 96 ]; then export P321_OUT=pat321/data; else export P321_OUT=pat329/lq$1; fi
for H in 128 256 512; do $PY pat329/topk.py --H $H > ${L}_H${H}.txt 2>&1 & done; wait
echo done

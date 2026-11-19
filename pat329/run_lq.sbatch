#!/bin/bash
#SBATCH --job-name=p329_lq
#SBATCH --output=pat329/logs/%j_lq.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=16:00:00
# PAT-329 LQ scan: retrain the FreshCast scorer with query window LQ=$1, then r-only / smap-fc / smap-classic (ctx 512)
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}_lq$1
export P321_LQ=$1 P321_OUT=pat329/lq$1
mkdir -p $P321_OUT; for f in train val test; do ln -sf ../../pat321/data/$f.pt $P321_OUT/$f.pt; done
for H in 128 256 512; do $PY pat321/fc_dysts.py --stage train --arm fc --H $H > ${L}_train_H${H}.txt 2>&1 & done; wait
for H in 128 256 512; do $PY pat329/lq_scan.py --H $H > ${L}_H${H}.txt 2>&1 & done; wait
echo done

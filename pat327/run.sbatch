#!/bin/bash
#SBATCH --job-name=p327_corr
#SBATCH --output=pat327/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# PAT-327: transform the retrieved continuations of FreshCast r-only (pair / path, learned) vs S-map; frozen PAT-321 fc scorer
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat327/logs/${SLURM_JOB_ID}
for H in 128 256 512; do for arm in pair path; do $PY pat327/fc_corr.py --stage train --arm $arm --H $H > ${L}_train_${arm}_H${H}.txt 2>&1 & done; done; wait
$PY pat327/fc_corr.py --stage test > ${L}_test.txt 2>&1
echo done

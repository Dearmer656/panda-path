#!/bin/bash
#SBATCH --job-name=p334_hid
#SBATCH --output=pat334/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
# PAT-334 G2 alone (signal vs hidden heads, ctx512)
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 P321_LQ=96 P321_OUT=pat321/data $PY pat334/mh_hid.py > pat334/logs/${SLURM_JOB_ID}_hid.txt 2>&1

echo done

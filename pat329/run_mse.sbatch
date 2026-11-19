#!/bin/bash
#SBATCH --job-name=p329_mse
#SBATCH --output=pat329/logs/%j_mse.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# PAT-329 MSE / amplitude check, one process per LQ on one GPU
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}_mse
P321_LQ=96 P321_OUT=pat321/data $PY pat329/mse_eval.py > ${L}96.txt 2>&1 &
P321_LQ=32 P321_OUT=pat329/lq32 $PY pat329/mse_eval.py > ${L}32.txt 2>&1 &
P321_LQ=64 P321_OUT=pat329/lq64 $PY pat329/mse_eval.py > ${L}64.txt 2>&1 &
wait; $PY pat329/mse_select.py > ${L}_select.txt 2>&1
echo done

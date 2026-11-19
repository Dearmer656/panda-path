#!/bin/bash
#SBATCH --job-name=p329_topk2
#SBATCH --output=pat329/logs/%j_topk2.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=16:00:00
# PAT-329 top-k: redo the 5 runs that hit CUDA OOM in job 51421 (same bs 64), in two waves on one GPU
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}_topk
run() { if [ $1 = 96 ]; then O=pat321/data; else O=pat329/lq$1; fi; P321_LQ=$1 P321_OUT=$O $PY pat329/topk.py --H $2 --bs 64 > ${L}$1_H$2.txt 2>&1; }
run 32 256 & run 32 512 & run 64 512 & wait
run 96 256 & run 96 512 & wait
echo done

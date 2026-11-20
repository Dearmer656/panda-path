#!/bin/bash
#SBATCH --job-name=p331_mem
#SBATCH --output=pat331/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
# PAT-331 pilot: memory attention in frozen Panda (RoPE), ctx 2048 origins, H128; check, then LB 6 and 8 packed on one GPU, then compare
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat331/logs/${SLURM_JOB_ID}; export P321_LQ=96 P321_CTX=2048 P330_BLOCKS=6 P330_HIST=ctx P321_OUT=pat330/fair2048
$PY pat331/mem.py --lb 6 --check > ${L}_check.txt 2>&1 || { echo check failed; exit 1; }
for LB in 6 8; do $PY pat331/mem.py --lb $LB > ${L}_lb$LB.txt 2>&1 & done; wait
$PY pat331/cmp.py > ${L}_cmp.txt 2>&1
echo done

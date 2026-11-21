#!/bin/bash
#SBATCH --job-name=p334_str
#SBATCH --output=pat334/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=16:00:00
# PAT-334 stride ablation: args = SETTING:STRIDE:SUB (SETTING full|ctx), anchors need a 512 window (AMIN=1)
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python
for a in "$@"; do IFS=: read st s sub <<< "$a"
  if [ $st = full ]; then ENV="P321_OUT=pat321/data P333_HIST=full"; else ENV="P321_CTX=2048 P321_OUT=pat321/data_ctx2048_path P333_HIST=ctx"; fi
  env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 P321_LQ=96 $ENV P334_STRIDE=$s P334_SUB=$sub P334_AMIN=1 $PY pat334/mh.py > pat334/logs/${SLURM_JOB_ID}_${st}_s${s}${sub}.txt 2>&1 &
done
wait
echo done

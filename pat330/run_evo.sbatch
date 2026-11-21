#!/bin/bash
#SBATCH --job-name=p330_evo
#SBATCH --output=pat330/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
# PAT-330 next task: segmented error evolution g beyond whole-window / final-segment distance
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat330/logs/${SLURM_JOB_ID}
P321_LQ=96 P321_OUT=pat321/data $PY pat330/evo.py pool > ${L}_evo_pool.txt 2>&1 &
P321_LQ=96 P330_BLOCKS=6 P321_OUT=pat330/blk $PY pat330/evo.py blk > ${L}_evo_blk.txt 2>&1 &
wait; $PY pat330/evo_cmp.py > ${L}_evo_cmp.txt 2>&1
echo done

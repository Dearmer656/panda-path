#!/bin/bash
#SBATCH --job-name=p330_vsel
#SBATCH --output=pat330/logs/%j_vsel.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# PAT-330 step 1: val selection over 4 scorer configs x {r_all, r_top1, smap_lambda}, then test vs parrot-full
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat330/logs/${SLURM_JOB_ID}; export P321_LQ=96
P321_OUT=pat321/data $PY pat330/valsel_pred.py --tag orig > ${L}_orig.txt 2>&1 &
P330_VALONLY=1 P321_OUT=pat329/valonly $PY pat330/valsel_pred.py --tag vonly > ${L}_vonly.txt 2>&1 &
P330_BLOCKS=6 P321_OUT=pat330/blk $PY pat330/valsel_pred.py --tag blk > ${L}_blk.txt 2>&1 &
P330_BLOCKS=6 P330_VALONLY=1 P321_OUT=pat330/blkv $PY pat330/valsel_pred.py --tag blkv > ${L}_blkv.txt 2>&1 &
wait; $PY pat330/valsel.py > ${L}_select.txt 2>&1
echo done

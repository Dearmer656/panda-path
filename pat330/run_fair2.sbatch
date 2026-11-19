#!/bin/bash
#SBATCH --job-name=p330_fair2
#SBATCH --output=pat330/logs/%j_fair2.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# PAT-330 step 3 redo of ctx 1024 (H256 scorer OOM in 51494, 8 trainings on one GPU)
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat330/logs/${SLURM_JOB_ID}; export P321_LQ=96 P330_BLOCKS=6 P330_HIST=ctx P321_CTX=1024 P321_OUT=pat330/fair1024
$PY pat321/fc_dysts.py --stage train --arm fc --H 256 > ${L}_train_1024_H256.txt 2>&1
P330_HS=128,256,512 $PY pat330/valsel_pred.py --tag fair1024 > ${L}_pred_1024.txt 2>&1
$PY pat330/fair_cmp.py > ${L}_cmp.txt 2>&1
echo done

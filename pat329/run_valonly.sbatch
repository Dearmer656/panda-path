#!/bin/bash
#SBATCH --job-name=p329_vonly
#SBATCH --output=pat329/logs/%j_vonly.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=16:00:00
# PAT-329 value-only readouts: (1) original scorer, shape entries masked; (2) scorer retrained on value entries only
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat329/logs/${SLURM_JOB_ID}_vonly; export P330_VALONLY=1 P321_LQ=96
mkdir -p pat329/valonly; for f in train val test; do ln -sf ../../pat321/data/$f.pt pat329/valonly/$f.pt; done
P321_OUT=pat321/data P329_TAG=_vmask $PY pat329/mse_eval.py > ${L}_vmask.txt 2>&1 &
for H in 128 256 512; do P321_OUT=pat329/valonly $PY pat321/fc_dysts.py --stage train --arm fc --H $H > ${L}_train_H${H}.txt 2>&1 & done; wait
P321_OUT=pat329/valonly P329_TAG=_vonly $PY pat329/mse_eval.py > ${L}_eval.txt 2>&1
$PY pat329/valonly_cmp.py > ${L}_cmp.txt 2>&1
echo done

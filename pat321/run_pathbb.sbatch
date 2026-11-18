#!/bin/bash
#SBATCH --job-name=p325_fc
#SBATCH --output=pat321/logs/%j_pathbb.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
# PAT-325: PAT-321 FreshCast pipeline unchanged, frozen backbone = PaTH (p318ut_path s99) instead of RoPE
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat321/logs/${SLURM_JOB_ID}_pathbb; export P321_OUT=pat321/data_pathbb; CK=pat318/ckpt/p318ut_path_s99_st100000-0/checkpoint-final; DU=pat321/data/test_dump.pkl
[ -f $P321_OUT/test.pt ] || $PY pat321/fc_dysts.py --stage prep --ckpt $CK --dump $DU > ${L}_prep.txt 2>&1 || exit 1
$PY pat321/fc_dysts.py --stage check --ckpt $CK --dump $DU > ${L}_check.txt 2>&1
for H in 128 256 512; do
  for arm in fc bil path; do $PY pat321/fc_dysts.py --stage train --ckpt $CK --arm $arm --H $H > ${L}_train_${arm}_H${H}.txt 2>&1 & done; wait
done
$PY pat321/fc_dysts.py --stage test --ckpt $CK --dump $DU > ${L}_test.txt 2>&1
echo done

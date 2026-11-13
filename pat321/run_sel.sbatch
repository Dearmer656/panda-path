#!/bin/bash
#SBATCH --job-name=p321_sel
#SBATCH --output=pat321/logs/%j_sel.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
# PAT-321 regularised re-run: for every arm and H, weight decay in {1e-2, 1e-1, 1} x early stopping on the held-out
# (val) training-objective loss; the (wd, epoch) with the lowest val loss is selected per arm, then tested once.
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat321/logs/${SLURM_JOB_ID}
for H in 128 256 512; do
  for wd in 1e-2 1e-1 1; do
    for arm in fc bil path; do $PY pat321/fc_dysts.py --stage train --arm $arm --H $H --wd $wd --tag _wd$wd > ${L}_train_${arm}_H${H}_wd$wd.txt 2>&1 & done; wait
  done
done
$PY pat321/fc_dysts.py --stage test --tags _wd1e-2,_wd1e-1,_wd1 --out_tag _sel > ${L}_test.txt 2>&1
echo done

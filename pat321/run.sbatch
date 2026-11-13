#!/bin/bash
#SBATCH --job-name=p321_fc
#SBATCH --output=pat321/logs/%j_run.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
# PAT-321 full run: prep (backbone pass + test windows) -> base check -> 3 arms x 3 horizons scorer training (3 at a time on 1 GPU) -> test
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat321/logs/${SLURM_JOB_ID}
[ -f pat321/data/test.pt ] || $PY pat321/fc_dysts.py --stage prep > ${L}_prep.txt 2>&1 || exit 1
$PY pat321/fc_dysts.py --stage check > ${L}_check.txt 2>&1
for H in 128 256 512; do
  for arm in fc bil path; do $PY pat321/fc_dysts.py --stage train --arm $arm --H $H > ${L}_train_${arm}_H${H}.txt 2>&1 & done; wait
done
$PY pat321/fc_dysts.py --stage test > ${L}_test.txt 2>&1
echo done

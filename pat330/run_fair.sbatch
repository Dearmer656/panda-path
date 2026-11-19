#!/bin/bash
#SBATCH --job-name=p330_fair
#SBATCH --output=pat330/logs/%j_fair.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# PAT-330 step 3: retrieval restricted to [t - ctx, t); blk scorer retrained per ctx; val selection; test vs backbones
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat330/logs/${SLURM_JOB_ID}; export P321_LQ=96 P330_BLOCKS=6 P330_HIST=ctx
for C in 512 1024 2048; do
  if [ $C = 512 ]; then SRC=pat321/data; HS=128,256; else SRC=pat321/data_ctx${C}_path; HS=128,256,512; fi
  mkdir -p pat330/fair$C; for f in train val test; do ln -sf ../../$SRC/$f.pt pat330/fair$C/$f.pt; done
  for H in ${HS//,/ }; do P321_CTX=$C P321_OUT=pat330/fair$C $PY pat321/fc_dysts.py --stage train --arm fc --H $H > ${L}_train_${C}_H${H}.txt 2>&1 & done
done; wait
for C in 512 1024 2048; do
  if [ $C = 512 ]; then HS=128,256; else HS=128,256,512; fi
  P321_CTX=$C P330_HS=$HS P321_OUT=pat330/fair$C $PY pat330/valsel_pred.py --tag fair$C > ${L}_pred_${C}.txt 2>&1 &
done; wait
$PY pat330/fair_cmp.py > ${L}_cmp.txt 2>&1
echo done

#!/bin/bash
#SBATCH --job-name=p330_blk
#SBATCH --output=pat330/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# PAT-330: 6 x 16-step segment errors as extra FreshCast descriptors; retrain (all entries / value only), evaluate, compare
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; L=pat330/logs/${SLURM_JOB_ID}; export P330_BLOCKS=6 P321_LQ=96
P321_OUT=pat321/data $PY pat330/test_blocks.py > ${L}_test.txt 2>&1 || { echo test failed; exit 1; }
for c in blk blkv; do mkdir -p pat330/$c; for f in train val test; do ln -sf ../../pat321/data/$f.pt pat330/$c/$f.pt; done; done
for H in 128 256 512; do
  P321_OUT=pat330/blk $PY pat321/fc_dysts.py --stage train --arm fc --H $H > ${L}_train_blk_H${H}.txt 2>&1 &
  P330_VALONLY=1 P321_OUT=pat330/blkv $PY pat321/fc_dysts.py --stage train --arm fc --H $H > ${L}_train_blkv_H${H}.txt 2>&1 &
done; wait
P321_OUT=pat330/blk P329_TAG=_blk $PY pat329/mse_eval.py > ${L}_eval_blk.txt 2>&1 &
P330_VALONLY=1 P321_OUT=pat330/blkv P329_TAG=_blkv $PY pat329/mse_eval.py > ${L}_eval_blkv.txt 2>&1 &
wait; $PY pat330/cmp.py > ${L}_cmp.txt 2>&1
echo done

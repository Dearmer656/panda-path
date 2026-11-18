#!/bin/bash
#SBATCH --job-name=p325_ctx
#SBATCH --output=pat321/logs/%j_ctx.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
# PAT-325 context scan: usage sbatch pat321/run_ctx.sbatch CTX. Records the official test windows at this context (same eval as
# pat318 ctx evals), then runs the unchanged PAT-321 FreshCast pipeline on the frozen PaTH and RoPE backbones at that context.
CTX=$1
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
PY=.venv/bin/python; export P321_CTX=$CTX; DU=pat321/data/test_dump_ctx$CTX.pkl
if [ ! -s $DU ]; then
  P321_DUMP=$DU P318_CTX=$CTX P318_LOAD_SEED=99 P318_EVAL_SCRIPT=pat321/dump_test.py bash pat318/eval.sh pat318/ckpt/p318ut_path_s99_st100000-0/checkpoint-final p325_dump_ctx$CTX 0 > pat321/logs/${SLURM_JOB_ID}_dump_ctx$CTX.txt 2>&1 || exit 1
fi
for BB in path rope; do
  [ $BB = path ] && CK=pat318/ckpt/p318ut_path_s99_st100000-0/checkpoint-final || CK=pat318/ckpt/p318ut_rope_s99_st100000-1/checkpoint-final
  export P321_OUT=pat321/data_ctx${CTX}_$BB; L=pat321/logs/${SLURM_JOB_ID}_ctx${CTX}_$BB
  [ -f $P321_OUT/test.pt ] || $PY pat321/fc_dysts.py --stage prep --ckpt $CK --dump $DU > ${L}_prep.txt 2>&1 || exit 1
  $PY pat321/fc_dysts.py --stage check --ckpt $CK --dump $DU > ${L}_check.txt 2>&1
  for H in 128 256 512; do
    for arm in fc bil path; do $PY pat321/fc_dysts.py --stage train --ckpt $CK --arm $arm --H $H > ${L}_train_${arm}_H${H}.txt 2>&1 & done; wait
  done
  $PY pat321/fc_dysts.py --stage test --ckpt $CK --dump $DU > ${L}_test.txt 2>&1
done
echo done

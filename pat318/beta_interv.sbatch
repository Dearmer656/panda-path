#!/bin/bash
#SBATCH --job-name=p318bi
#SBATCH --output=pat318/logs/%j_beta_interv.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# PAT-318 beta intervention on the frozen PaTH checkpoint (PatchTST-univariate path, seed 99, trained at ctx 512; ctx 1024 /
# 2048 are length extrapolation of that same checkpoint). Official eval (pat318/eval.sh), beta changed by pat318/beta_interv.py.
# usage: sbatch pat318/beta_interv.sbatch CTX   (one 6000 per ctx: c in {1, 0, 0.5, 1.5} and shuffle seeds {0, 1, 2}, run in sequence)
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
CK=pat318/ckpt/p318ut_path_s99_st100000-0/checkpoint-final
T=()
for iv in c1 c0 c0.5 c1.5 s0 s1 s2; do T+=("$iv $1"); done
run() { iv=$1; ctx=$2; gpu=$3; name=p318ut_path_s99_st100000_ls99_bi${iv}_ctx$ctx
  if [ ${iv:0:1} = c ]; then E="P318_BETA_C=${iv:1}"; else E="P318_BETA_SHUF=${iv:1}"; fi
  env $E P318_BETA_STATS=pat318/eval/$name/beta_stats.json P318_LOAD_SEED=99 P318_CTX=$ctx P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK $name $gpu > pat318/logs/${SLURM_JOB_ID}_bi_$name.txt 2>&1
  echo "$(date +%T) done $name exit $?"; }
for t in "${T[@]}"; do run $t 0; done
wait; echo all done

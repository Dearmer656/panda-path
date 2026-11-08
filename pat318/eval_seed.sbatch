#!/bin/bash
#SBATCH --job-name=p318_evseed
#SBATCH --output=pat318/logs/%j_evseed.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# PAT-318: released checkpoint, official eval, torch RNG seeded before model construction (99 = training seed, 1234 = control).
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
for S in 99 1234; do
  P318_LOAD_SEED=$S P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh data/ckpt_panda released_load$S 0 > pat318/logs/${SLURM_JOB_ID}_eval_load$S.txt 2>&1
done
echo done

#!/bin/bash
#SBATCH --job-name=p318_evrel
#SBATCH --output=pat318/logs/%j_evrel.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# PAT-318: released Panda checkpoint, official eval on base40 + skew40 test_zeroshot (9347 trajectories).
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
bash pat318/eval.sh data/ckpt_panda released_full 0 > pat318/logs/${SLURM_JOB_ID}_eval_released_full.txt 2>&1
echo done

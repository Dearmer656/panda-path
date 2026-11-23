#!/bin/bash
#SBATCH --job-name=p335_ev
#SBATCH --output=pat335/logs/%j_ev.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
# PAT-335 official eval of a checkpoint with load seed 99 (same as p318_*_ls99). usage: sbatch pat335/eval_ls99.sbatch CKPT NAME
CK=$(ls -d $1 | tail -1); NAME=$2; echo "ckpt $CK"; [ -d "$CK" ] || exit 1
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${NAME}_ls99_ctx$1 $2 > pat335/logs/${SLURM_JOB_ID}_eval_${NAME}_ls99_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & ev 2048 2 & wait
echo done

#!/bin/bash
#SBATCH --job-name=p341_eval0
#SBATCH --output=pat341/logs/%j_eval0.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-341 step 1 (no training): eval CK (default: p318_path final with layer 0 transport removed = nope1_path) at ctx 512 / 1024 / 2048, ls99.
CK=${1:-pat318/ckpt/p341_nope1_from_path_final}; NAME=${2:-p341_nope1_from_path_final}
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor
for c in 512 1024 2048; do P318_LOAD_SEED=99 P318_CTX=$c P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${NAME}_ls99_ctx$c 0 > pat341/logs/${SLURM_JOB_ID}_eval_${NAME}_ctx$c.txt 2>&1; echo "ctx$c exit $?"; done

#!/bin/bash
#SBATCH --job-name=p336_ev512
#SBATCH --output=pat336/logs/%j_ev512.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# official eval (load seed 99) of one checkpoint at ctx 512 only. usage: sbatch pat336/eval512.sbatch CKPT NAME
CK=$1; NAME=$2; echo "ckpt $CK"; [ -d "$CK" ] || exit 1
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor
P318_LOAD_SEED=99 P318_CTX=512 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${NAME}_ls99_ctx512 0 > pat336/logs/${SLURM_JOB_ID}_eval_${NAME}.txt 2>&1
echo done

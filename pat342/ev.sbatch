#!/bin/bash
#SBATCH --job-name=p342_ev4k
#SBATCH --output=pat342/logs/%j_ev.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm54,elm55
# PAT-342: ls99 eval (ctx 512/1024/2048) of one checkpoint, then Lie analysis of the given ckpts. usage: sbatch pat342/ev.sbatch CK NAME CKPT...
CK=$1; NAME=$2; shift 2; [ -d "$CK" ] || exit 1
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${NAME}_ls99_ctx$1 $2 > pat342/logs/${SLURM_JOB_ID}_eval_${NAME}_ctx$1.txt 2>&1; }
if [ -n "$SEQ" ]; then ev 512 0; CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pat342.what_lie pat342/out/$NAME "$@" 2>&1 | grep -vE "Some weights|IS expected|IS NOT" & ev 1024 0; ev 2048 0; wait; else (ev 512 0; CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pat342.what_lie pat342/out/$NAME "$@" 2>&1 | grep -vE "Some weights|IS expected|IS NOT") & ev 1024 1 & ev 2048 2 & wait; fi
echo done

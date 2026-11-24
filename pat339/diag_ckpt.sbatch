#!/bin/bash
#SBATCH --job-name=p339_diag
#SBATCH --output=pat339/logs/%j_diag.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-335: diagnostics v2 (ctx 512/1024/2048) + fp32 vs fp64 numerics + ls99 eval at ctx 512 of one intermediate checkpoint.
# usage: sbatch pat339/diag_ckpt.sbatch CKPT NAME
CK=$1; NAME=$2; echo "ckpt $CK"; [ -d "$CK" ] || exit 1
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4
.venv/bin/python -m pat336.props pat339/out/$NAME $CK 2>&1 | grep -E "^==|^ *L|FLAG|Error|Traceback"
.venv/bin/python -m pat336.numcheck $CK 2>&1 | grep -vE "Some weights|IS expected|IS NOT"
P318_LOAD_SEED=99 P318_CTX=512 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${NAME}_ls99_ctx512 0 > pat339/logs/${SLURM_JOB_ID}_eval_${NAME}.txt 2>&1
echo done

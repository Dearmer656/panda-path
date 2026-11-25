#!/bin/bash
#SBATCH --job-name=p342_ev1
#SBATCH --output=pat342/logs/%j_ev1.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm54,elm55
# PAT-342: one ls99 eval of CK at CTX under the env in force (P342_LIE_INTV, P318_LOGN pass through). usage: sbatch pat342/ev1.sbatch CK NAME CTX
CK=$1; NAME=$2; CTX=$3; [ -d "$CK" ] || exit 1; echo "intv=$P342_LIE_INTV logn=$P318_LOGN"
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4
P318_LOAD_SEED=99 P318_CTX=$CTX P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${NAME}_ls99_ctx$CTX 0 > pat342/logs/${SLURM_JOB_ID}_eval_${NAME}_ctx$CTX.txt 2>&1
tail -2 pat342/logs/${SLURM_JOB_ID}_eval_${NAME}_ctx$CTX.txt; echo done

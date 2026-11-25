#!/bin/bash
#SBATCH --job-name=p342_lie
#SBATCH --output=pat342/logs/%j_what_lie.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm54,elm55
# PAT-342: Lie-transform analysis (pat342/what_lie.py) of early checkpoints. usage: sbatch pat342/what_lie.sbatch OUT CKPT [CKPT ...]
OUT=$1; shift
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4
.venv/bin/python -m pat342.what_lie pat342/out/$OUT "$@" 2>&1 | grep -vE "Some weights|IS expected|IS NOT"
echo done

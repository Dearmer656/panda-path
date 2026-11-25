#!/bin/bash
#SBATCH --job-name=p342_intv
#SBATCH --output=pat342/logs/%j_intv.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm54,elm55
# PAT-342 review tasks 2+3: pat342/intv_attn.py on 10k (all interventions) and on init / 4k (none only), ctx 512 / 2048.
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4 P336_CTXS=512,2048
C=pat318/ckpt/p339_nope1_lie_s99_st100000-0; F="grep -vE Some.weights|IS.expected|IS.NOT"
.venv/bin/python -m pat342.intv_attn pat342/out/n1lie_10k $C/checkpoint-10000 2>&1 | grep -vE "Some weights|IS expected|IS NOT"
P342_INTVS=none .venv/bin/python -m pat342.intv_attn pat342/out/n1lie_init_4k pat318/ckpt/p342_nope1_lie_init_s99 $C/checkpoint-4000 2>&1 | grep -vE "Some weights|IS expected|IS NOT"
echo done

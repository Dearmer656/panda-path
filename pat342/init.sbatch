#!/bin/bash
#SBATCH --job-name=p342_init
#SBATCH --output=pat342/logs/%j_init.txt
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm54,elm55
# PAT-342 control A: Lie statistics of the untrained (seed 99 init) nope1_lie model.
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4
D=pat318/ckpt/p342_nope1_lie_init_s99
.venv/bin/python pat342/make_init.py pat318/ckpt/p339_nope1_lie_s99_st100000-0/checkpoint-1000 $D || exit 1
.venv/bin/python -m pat342.what_lie pat342/out/p342_nope1_lie_init $D 2>&1 | grep -vE "Some weights|IS expected|IS NOT"
echo done

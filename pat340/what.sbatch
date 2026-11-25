#!/bin/bash
#SBATCH --job-name=p340_what
#SBATCH --output=pat340/logs/%j_what.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-340: what the learned transitions do (pat340/what.py) at ctx 512 / 1024 / 2048, 48 held-out systems, load seed 99.
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export P336_CTXS=512,1024,2048 P336_NSYS=48
.venv/bin/python -m pat340.what pat340/out/m $(ls -d pat318/ckpt/p337_path2_s99_st100000-0/checkpoint-* | sort -t- -k2 -n | tail -1) pat318/ckpt/p318_path_s99_st100000-0/checkpoint-final pat318/ckpt/p318_nope_s99_st100000-0/checkpoint-final pat318/ckpt/p318_rope_s99_st100000-0/checkpoint-final pat318/ckpt/p318_rope_path_s99_st100000-0/checkpoint-final pat318/ckpt/p338_rope1_path_s99_st100000-0/checkpoint-50000 pat318/ckpt/p335_path_shared_s99_st100000-0/checkpoint-50000 2>&1 | grep -vE "Some weights|IS expected|IS NOT|TRAIN this|^- "
echo done

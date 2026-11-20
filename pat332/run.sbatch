#!/bin/bash
#SBATCH --job-name=p332_wret
#SBATCH --output=pat332/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
# PAT-332 Wasserstein retrieval vs Euclid (ctx 512, top-1 copy, test)
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
P321_LQ=96 P321_OUT=pat321/data .venv/bin/python pat332/wret.py

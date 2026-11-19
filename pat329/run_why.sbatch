#!/bin/bash
#SBATCH --job-name=p329_why
#SBATCH --output=pat329/logs/%j_why.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=6:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
P321_LQ=96 P321_OUT=pat321/data .venv/bin/python pat329/why512.py

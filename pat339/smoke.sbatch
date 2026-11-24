#!/bin/bash
#SBATCH --job-name=p339_smoke
#SBATCH --output=pat339/logs/%j_smoke.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-339 smoke: ARM (rope1_lie_s / rope1_lie_k / rope1_path), 60 steps on 1 x 3090, checkpoints at 20 / 40 (+ final).
# Purpose: check the arm trains / saves / reloads, and give mixed-layer checkpoints for validating pat336/props.py. Not evidence.
ARM=$1; SEED=99
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4 P318_SAVE_AT=20,40
P318_PREFIX=p339smoke bash pat318/train.sh $ARM $SEED 60 0 train.per_device_train_batch_size=128 train.gradient_accumulation_steps=1 train.dataloader_num_workers=6 > pat339/logs/${SLURM_JOB_ID}_train_${ARM}.txt 2>&1
ls pat318/ckpt/p339smoke_${ARM}_s${SEED}_st60-*/

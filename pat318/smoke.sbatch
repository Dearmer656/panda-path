#!/bin/bash
#SBATCH --job-name=p318_smoke
#SBATCH --output=pat318/logs/%j_smoke.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:2
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
# PAT-318 smoke: (a) official eval of the released checkpoint (GPU 0, in background);
# (b) 300 training steps per arm on 1 GPU each (per-device bs 1024 = the released per-device setting) to time s/step.
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
nvidia-smi --query-gpu=name,memory.total --format=csv
bash pat318/eval.sh data/ckpt_panda released 0 > pat318/logs/${SLURM_JOB_ID}_eval_released.txt 2>&1 &
# 2 GPUs: arms run one after another on each GPU so the timing is not shared
( for ARM in rope path; do bash pat318/train.sh $ARM 99 300 0 train.output_dir=pat318/ckpt_smoke train.dataloader_num_workers=6 > pat318/logs/${SLURM_JOB_ID}_smoke_${ARM}.txt 2>&1; done ) &
( for ARM in nope rope_path; do bash pat318/train.sh $ARM 99 300 1 train.output_dir=pat318/ckpt_smoke train.dataloader_num_workers=6 > pat318/logs/${SLURM_JOB_ID}_smoke_${ARM}.txt 2>&1; done ) &
wait
echo done

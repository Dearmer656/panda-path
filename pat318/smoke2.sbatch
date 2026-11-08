#!/bin/bash
#SBATCH --job-name=p318_smoke2
#SBATCH --output=pat318/logs/%j_smoke2.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# PAT-318 smoke (part 2): 300 steps per arm on one GPU, sequential, to time s/step for nope / path / rope_path.
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
rm -rf pat318/ckpt_smoke
for ARM in nope path rope_path; do
  bash pat318/train.sh $ARM 99 300 0 train.output_dir=pat318/ckpt_smoke train.dataloader_num_workers=6 > pat318/logs/${SLURM_JOB_ID}_smoke_${ARM}.txt 2>&1
done
echo done

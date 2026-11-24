#!/bin/bash
#SBATCH --job-name=p336_bench
#SBATCH --output=pat336/logs/%j_bench.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3:00:00
# per-step time of the real training (1 x 3090, per-device 128, no compile, 60 steps): path vs lie r=4 vs lie r=2
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor
for cfg in "path 4" "lie 4" "lie 2"; do set -- $cfg
  P336_LIE_RANK=$2 P318_PREFIX=bench$2 bash pat318/train.sh $1 99 60 0 train.per_device_train_batch_size=128 train.dataloader_num_workers=6 train.torch_compile=false train.save_steps=1000000 train.output_dir=pat336/bench_ckpt > pat336/logs/${SLURM_JOB_ID}_bench_$1_r$2.txt 2>&1
  echo "$1 r$2: $(tr '\r' '\n' < pat336/logs/${SLURM_JOB_ID}_bench_$1_r$2.txt | grep -E 'it/s|s/it' | tail -1 | cut -c1-90)"
done

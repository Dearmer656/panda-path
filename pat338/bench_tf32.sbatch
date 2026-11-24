#!/bin/bash
#SBATCH --job-name=p338_bench
#SBATCH --output=pat338/logs/%j_bench.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# per-step time, 1 x 3090, per-device 128, no compile, 60 steps: path fp32 (again, same node) vs path tf32
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
for tf in false true; do
  P318_PREFIX=benchtf$tf bash pat318/train.sh path 99 60 0 train.per_device_train_batch_size=128 train.dataloader_num_workers=6 train.torch_compile=false train.tf32=$tf train.save_steps=1000000 train.output_dir=pat338/bench_ckpt > pat338/logs/${SLURM_JOB_ID}_bench_tf32_$tf.txt 2>&1
  echo "path tf32=$tf: $(tr '\r' '\n' < pat338/logs/${SLURM_JOB_ID}_bench_tf32_$tf.txt | grep -E 'it/s|s/it' | sed -n '58p' | cut -c1-80)"
done
rm -rf pat338/bench_ckpt

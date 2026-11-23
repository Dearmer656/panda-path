#!/bin/bash
#SBATCH --job-name=p336_lie
#SBATCH --output=pat336/logs/%j_full.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4-04:00:00
# PAT-336 lie (Lie-exponential transport, rank 4), official recipe as p318_path_s99 (global batch 4096 = 2 GPUs x 512 x
# grad-accum 4, 100k steps, seed 99, checkpoints 50k / 100k); then official eval (load seed 99) at ctx 512 / 1024 / 2048.
ARM=lie; SEED=99
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
RUN=p336_${ARM}_s${SEED}_st100000
P318_PREFIX=p336 bash pat318/train.sh $ARM $SEED 100000 0,1 train.per_device_train_batch_size=512 train.gradient_accumulation_steps=4 train.dataloader_num_workers=6 > pat336/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls99_ctx$1 $2 > pat336/logs/${SLURM_JOB_ID}_eval_${RUN}_ls99_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & wait; ev 2048 0
echo done

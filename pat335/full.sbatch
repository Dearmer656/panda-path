#!/bin/bash
#SBATCH --job-name=p335_full
#SBATCH --output=pat335/logs/%j_full.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
# PAT-335 path_shared, official recipe identical to pat318/full.sbatch (global batch 4096 = 4 GPUs x 512 x grad-accum 2,
# 100k steps, seed 99, checkpoints at 50k / 100k); then official eval at ctx 512 / 1024 / 2048 (same as p318_path_s99).
ARM=path_shared; SEED=99
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
RUN=p335_${ARM}_s${SEED}_st100000
P318_PREFIX=p335 bash pat318/train.sh $ARM $SEED 100000 0,1,2,3 train.per_device_train_batch_size=512 train.gradient_accumulation_steps=2 train.dataloader_num_workers=6 > pat335/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls99_ctx$1 $2 > pat335/logs/${SLURM_JOB_ID}_eval_${RUN}_ls99_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & ev 2048 2 & wait
echo done

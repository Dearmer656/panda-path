#!/bin/bash
#SBATCH --job-name=p336b_lie
#SBATCH --output=pat336/logs/%j_full.txt
#SBATCH --partition=lang_gpu_week
#SBATCH --account=lang
#SBATCH --qos=qos_lang
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=8-00:00:00
# PAT-336 lie (Lie-exponential transport, rank 4, transpose convention, Gram-Schmidt), official recipe as p318_path_s99
# (global batch 4096 = 4 GPUs x 256 x grad-accum 4, 100k steps, seed 99, checkpoints 50k / 100k); then ls99 eval at ctx 512 / 1024 / 2048.
# bench (1 x 3090, 128/device): path 0.28 s/it, lie r4 1.22 s/it -> expected ~4-5 s/step here, ~5 days.
ARM=lie; SEED=99
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4
RUN=p336b_${ARM}_s${SEED}_st100000
P318_PREFIX=p336b bash pat318/train.sh $ARM $SEED 100000 0,1,2,3 train.per_device_train_batch_size=256 train.gradient_accumulation_steps=4 train.dataloader_num_workers=6 > pat336/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls99_ctx$1 $2 > pat336/logs/${SLURM_JOB_ID}_eval_${RUN}_ls99_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & ev 2048 2 & wait
echo done

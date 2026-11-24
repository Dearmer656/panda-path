#!/bin/bash
#SBATCH --job-name=p338_rope1lie
#SBATCH --output=pat338/logs/%j_full.txt
#SBATCH --partition=lang_gpu_week
#SBATCH --account=lang
#SBATCH --qos=qos_lang
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=8-00:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-338 rope1_lie: layer 0 temporal attention RoPE (as official Panda), layers 1..7 lie (rank 4, as p336b_lie_s99).
# (global batch 4096 = 4 GPUs x 256 x grad-accum 4, 100k steps, seed 99, checkpoints 50k / 100k); then ls99 eval at ctx 512 / 1024 / 2048.
# bench (1 x 3090, 128/device): path 0.28 s/it, lie r4 1.22 s/it -> expected ~4-5 s/step here, ~5 days.
ARM=rope1_lie; SEED=99
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=4
RUN=p338_${ARM}_s${SEED}_st100000
P318_PREFIX=p338 bash pat318/train.sh $ARM $SEED 100000 0,1,2,3 train.per_device_train_batch_size=256 train.gradient_accumulation_steps=4 train.dataloader_num_workers=6 > pat338/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls99_ctx$1 $2 > pat338/logs/${SLURM_JOB_ID}_eval_${RUN}_ls99_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & ev 2048 2 & wait
echo done

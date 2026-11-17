#!/bin/bash
#SBATCH --job-name=p318m3
#SBATCH --output=pat318/logs/%j_base.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
# PAT-318 baseline reproduction: usage P318_CHATTN=false|true P318_PREFIX=p318ut|p318 sbatch --export=ALL pat318/base.sbatch ARM SEED (eval with load seed = train seed)
# 4x A100-40GB variant: 4 GPUs x per-device 512 x grad-accum 2 = global batch 4096, same 100k steps / lr schedule (mamba3 needs ~48GB at 1024).
# Then official eval at ctx 512 / 1024 / 2048.
ARM=$1; SEED=$2
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
RUN=${P318_PREFIX:-p318}_${ARM}_s${SEED}_st100000
bash pat318/train.sh $ARM $SEED 100000 0,1,2,3 train.per_device_train_batch_size=512 train.gradient_accumulation_steps=2 train.dataloader_num_workers=6 $P318_EXTRA > pat318/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=$SEED P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls${SEED}_ctx$1 $2 > pat318/logs/${SLURM_JOB_ID}_eval_${RUN}_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & wait
ev 2048 0
echo done

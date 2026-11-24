#!/bin/bash
#SBATCH --job-name=p338_rope1
#SBATCH --output=pat338/logs/%j_full.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --qos=qos_lang
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=4-04:00:00
# PAT-338 rope1_path: layer 0 temporal attention RoPE (as official Panda), layers 1..7 PaTH (as p318_path_s99).
# Official recipe as p318_path_s99 (global batch 4096 = 4 x A100 x 512 x grad-accum 2, 100k steps, seed 99, ckpt 50k / 100k);
# then ls99 eval at ctx 512 / 1024 / 2048.
ARM=rope1_path; SEED=99
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor
RUN=p338_${ARM}_s${SEED}_st100000
P318_PREFIX=p338 bash pat318/train.sh $ARM $SEED 100000 0,1,2,3 train.per_device_train_batch_size=512 train.gradient_accumulation_steps=2 train.dataloader_num_workers=6 > pat338/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls99_ctx$1 $2 > pat338/logs/${SLURM_JOB_ID}_eval_${RUN}_ls99_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & ev 2048 2 & wait
echo done

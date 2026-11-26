#!/bin/bash
#SBATCH --job-name=p339_lie
#SBATCH --output=pat339/logs/%j_full.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-339 Lie ablation (PAT-335 item 3): ARM in rope1_lie_s (K=0) / rope1_lie_k (lambda=0) / rope1_lie (KS, rerun with intermediate ckpts).
# Same recipe as p338_rope1_lie_s99 (layer 0 RoPE, layers 1..7 lie rank 4; global batch 4096 = 4 x 256 x accum 4; 100k steps, seed 99);
# Env GPUS (default 0,1,2,3) / BS / ACC / NW / EXTRA (extra hydra overrides, e.g. train.torch_compile=false on Pascal) override; GPUS x BS x ACC must stay 4096.
# Env P336_LIE_RANK (default 4) / PFX (run prefix, default p339; PAT-342 r = 2 run: PFX=p342r2 P336_LIE_RANK=2). Evals of a rank-2 ckpt need P336_LIE_RANK=2 too.
# ckpts 5k, 10k, ..., 90k, final. BS x ACC x 4 must stay 4096; RESUME=ckpt dir to continue a run. Partition / gres given on the sbatch command line (A100 x 4). Then ls99 eval of final at ctx 512/1024/2048.
ARM=$1; SEED=99
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor P336_LIE_RANK=${P336_LIE_RANK:-4}
RUN=${PFX:-p339}_${ARM}_s${SEED}_st100000
P318_PREFIX=${PFX:-p339} bash pat318/train.sh $ARM $SEED 100000 ${GPUS:-0,1,2,3} train.per_device_train_batch_size=${BS:-256} train.gradient_accumulation_steps=${ACC:-4} train.dataloader_num_workers=${NW:-6} ${EXTRA} ${RESUME:+train.resume_from_checkpoint=$RESUME} > pat339/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls99_ctx$1 $2 > pat339/logs/${SLURM_JOB_ID}_eval_${RUN}_ls99_ctx$1.txt 2>&1; }
NG=$(echo ${GPUS:-0,1,2,3} | tr , "\n" | wc -l); ev 512 0 & ev 1024 $((1 % NG)) & ev 2048 $((2 % NG)) & wait
echo done

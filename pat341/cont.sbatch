#!/bin/bash
#SBATCH --job-name=p341_cont
#SBATCH --output=pat341/logs/%j_cont.txt
#SBATCH --partition=gpu_long
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-341: warm start from p318_nope_s99 checkpoint-final, continue 10k steps with ARM = path (all 8 layers, new w_proj /
# beta_proj, beta_proj weight 0 bias -3 -> beta 0.095) or ARM = nope (control, identical schedule). Global batch 4096
# (GPUS x BS x ACC), peak lr 1e-4, cosine, warmup 10 %, seed 99, ckpts 5k + final. Gres / cpus on the sbatch line.
# Env INIT / STEPS / PFX / SAVE_AT override the start checkpoint, steps, run prefix, extra ckpts (p341p: start from p318_path final).
# Then ls99 eval of final at ctx 512 / 1024 / 2048.
ARM=$1; SEED=99; NGPU=${NGPU:-4}
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export TRITON_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/triton TORCHINDUCTOR_CACHE_DIR=/project/nlp-work5/hongyu-s/.cache/inductor
STEPS=${STEPS:-10000}; PFX=${PFX:-p341}
export P341_INIT_FROM=${INIT:-pat318/ckpt/p318_nope_s99_st100000-0/checkpoint-final} P318_SAVE_AT=${SAVE_AT:-5000}
RUN=${PFX}_${ARM}_s${SEED}_st${STEPS}
GPUS=$(seq -s, 0 $((NGPU - 1)))
P318_PREFIX=$PFX bash pat318/train.sh $ARM $SEED $STEPS $GPUS train.learning_rate=1e-4 train.per_device_train_batch_size=${BS:-256} train.gradient_accumulation_steps=${ACC:-4} train.dataloader_num_workers=${NW:-2} > pat341/logs/${SLURM_JOB_ID}_train_${RUN}.txt 2>&1
CK=$(ls -d pat318/ckpt/${RUN}-*/checkpoint-final | tail -1)
echo "checkpoint: $CK"
[ -d "$CK" ] || exit 1
ev() { P318_LOAD_SEED=99 P318_CTX=$1 P318_EVAL_SCRIPT=pat318/evaluate_wrap.py bash pat318/eval.sh $CK ${RUN}_ls99_ctx$1 $2 > pat341/logs/${SLURM_JOB_ID}_eval_${RUN}_ls99_ctx$1.txt 2>&1; }
ev 512 0 & ev 1024 1 & ev 2048 $((2 % NGPU)) & wait
echo done

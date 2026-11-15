#!/bin/bash
# PAT-318: train Panda-21M with the released checkpoint's exact config (data/ckpt_panda/training_info.json), only temporal_pos varies.
# Usage: bash pat318/train.sh ARM SEED MAX_STEPS GPUS(e.g. 0,1,2,3) [extra hydra overrides...]
ARM=$1; SEED=$2; STEPS=$3; GPUS=$4; shift 4
cd "$(dirname "$0")/.."
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PORT=$((29500 + RANDOM % 2000))
export WANDB_MODE=disabled PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
ulimit -n 99999
CUDA_VISIBLE_DEVICES=$GPUS .venv/bin/torchrun --nproc-per-node $NG --master-port $PORT scripts/panda/train.py \
    run_name=${P318_PREFIX:-p318}_${ARM}_s${SEED}_st${STEPS} \
    wandb.log=false \
    train_data_dirs='["data/final_skew40/train","data/final_base40/train"]' \
    shuffle_buffer_length=100_000 min_past=60 max_missing_prop=0.9 fixed_dim=3 \
    augmentations.augmentation_rate=0.2 \
    'augmentations.probabilities=[0.3333333333333333,0.3333333333333333,0.3333333333333333,0.0,0.0]' \
    patchtst.mode=predict patchtst.context_length=512 patchtst.prediction_length=128 \
    patchtst.patch_length=16 patchtst.patch_stride=16 patchtst.num_hidden_layers=8 patchtst.d_model=512 \
    patchtst.num_attention_heads=8 patchtst.ffn_dim=512 patchtst.channel_attention=${P318_CHATTN:-true} patchtst.norm_type=rmsnorm \
    patchtst.pooling_type=mean patchtst.loss=mse patchtst.distribution_output=null \
    patchtst.max_wavelength=500 patchtst.rope_percent=0.75 patchtst.channel_rope=false \
    patchtst.use_dynamics_embedding=true patchtst.num_poly_feats=120 patchtst.poly_degrees=2 \
    patchtst.rff_trainable=false patchtst.rff_scale=1.0 patchtst.num_rff=256 \
    patchtst.temporal_pos=${P318_POS:-$ARM} \
    train.seed=$SEED train.max_steps=$STEPS train.save_steps=50_000 train.log_steps=100 \
    train.per_device_train_batch_size=1024 train.learning_rate=1e-3 train.lr_scheduler_type=cosine \
    train.warmup_ratio=0.1 train.weight_decay=0.0 train.max_grad_norm=1.0 train.tf32=false train.torch_compile=true \
    train.dataloader_num_workers=16 train.dataloader_prefetch_factor=2 \
    train.output_dir=pat318/ckpt \
    "$@"

#!/bin/bash
# PAT-318: official Panda evaluation (scripts/panda/run_eval.sh settings), skew40 test_zeroshot (base40 test is not released).
# Usage: bash pat318/eval.sh CKPT_DIR OUT_NAME GPU [extra hydra overrides...]
CK=$1; OUT=$2; GPU=$3; shift 3
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled PYTHONWARNINGS=ignore
ulimit -n 99999
.venv/bin/python scripts/panda/evaluate.py \
    eval.mode=predict eval.checkpoint_path=$CK \
    eval.data_paths_lst='["data/final_skew40/test_zeroshot"]' \
    eval.num_subdirs=null eval.num_test_instances=6 eval.window_style=sampled \
    eval.batch_size=64 eval.context_length=512 eval.prediction_length=512 eval.limit_prediction_length=false \
    eval.metrics_save_dir=pat318/eval/$OUT eval.metrics_fname=metrics eval.overwrite=true \
    eval.device=cuda:$GPU eval.save_labels=false eval.save_forecasts=false eval.seed=99 \
    "$@"

#!/bin/bash
# PAT-318 mamba3 arm smoke: unit checks, 60 training steps on 1 GPU (compile on), checkpoint round trip, peak memory.
cd "$(dirname "$0")/.."
.venv/bin/python -m pat318.test_mamba3 2>&1 | grep -v -i warn
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader >> pat318/logs/smoke_mamba3_mem.txt; sleep 2; done ) & MON=$!
P318_CHATTN=false P318_PREFIX=p318smoke bash pat318/train.sh mamba3 99 60 0 train.per_device_train_batch_size=1024 train.dataloader_num_workers=4 train.output_dir=pat318/ckpt_smoke > pat318/logs/smoke_mamba3_train.txt 2>&1
echo "train exit $?"; kill $MON
grep -E "trainable parameters|Error|error" pat318/logs/smoke_mamba3_train.txt | head -5
tr '\r' '\n' < pat318/logs/smoke_mamba3_train.txt | grep -E "^ *[0-9]+%" | tail -3
echo "peak mem $(sort -n pat318/logs/smoke_mamba3_mem.txt | tail -1)"
CK=$(ls -d pat318/ckpt_smoke/p318smoke_mamba3_s99_st60-*/checkpoint-final | tail -1); echo "ckpt $CK"
.venv/bin/python -m pat318.test_mamba3 $CK 2>&1 | grep -v -i warn | grep "^5"

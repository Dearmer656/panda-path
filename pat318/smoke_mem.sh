#!/bin/bash
# PAT-318: peak GPU memory of the fox arm on one GPU at a given per-device batch (usage: bash pat318/smoke_mem.sh BS)
cd "$(dirname "$0")/.."
( while true; do nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> pat318/logs/smoke_mem_$1.txt; sleep 2; done ) & MON=$!
P318_CHATTN=false P318_PREFIX=p318smoke bash pat318/train.sh fox 99 30 0 train.per_device_train_batch_size=$1 train.dataloader_num_workers=4 train.output_dir=pat318/ckpt_smoke > pat318/logs/smoke_train_$1.txt 2>&1
kill $MON; echo "exit $?"; sort -n pat318/logs/smoke_mem_$1.txt | tail -1; tail -c 300 pat318/logs/smoke_train_$1.txt | tr '\r' '\n' | tail -3

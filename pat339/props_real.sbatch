#!/bin/bash
#SBATCH --job-name=p339_props
#SBATCH --output=pat339/logs/%j_props.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74,elm41,elm42,elm53,elm55
# PAT-335 item 1/4: diagnostics v2 + numerics on existing trained checkpoints (seed 99): path final / 50k, rope_path final.
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
C=pat318/ckpt; P=$C/p318_path_s99_st100000-0; RP=$C/p318_rope_path_s99_st100000-0
.venv/bin/python -m pat336.props pat339/out/trained $P/checkpoint-final $P/checkpoint-50000 $RP/checkpoint-final
.venv/bin/python -m pat336.numcheck $P/checkpoint-final $RP/checkpoint-final
for c in 512 1024 2048; do .venv/bin/python -m pat336.assoc pat339/out/trained_comm.csv $P/checkpoint-final $c pat318/eval/p318_path_s99_ls99_ctx$c; done
.venv/bin/python -m pat336.jac pat339/out/trained pat339/out/trained_comm.csv
echo done

#!/bin/bash
cd /project/nlp-work5/hongyu-s/state_path_attn/panda
export P321_OUT=pat321/smoke P321_MAX_TEST=300 P321_EPOCHS=2
PY=.venv/bin/python
$PY pat321/fc_dysts.py --stage prep --dump pat321/data/test_dump.pkl --n_train_traj 60 --n_val_traj 20 && $PY pat321/fc_dysts.py --stage check && for arm in fc bil path; do for H in 128 512; do $PY pat321/fc_dysts.py --stage train --arm $arm --H $H --n_q 2000 || exit 1; done; done

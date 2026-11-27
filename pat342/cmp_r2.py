"""PAT-342: nope1_lie r = 2 (p342r2) vs r = 4 (nope1_lie) and nope1_path at matched steps, median sMAPE / MSE, paired bootstrap 95% CI
(300 resamples, seed 0). Pre-registered main criterion (PAT-335 a199fde8): ctx2048 L128 sMAPE, r2 - r4 < 0 with CI excluding 0, at 10k and 20k.
usage: python pat342/cmp_r2.py STEP(10k|20k) CTX"""
import sys, os; sys.path.insert(0, '.')
import numpy as np
from pat318.agg import load
s, c = sys.argv[1], int(sys.argv[2]); E = 'pat318/eval'; rng = np.random.default_rng(0)
A = f'{E}/p342r2_{s}_ls99_ctx{c}'
r4 = 'p342_n1lie10k_none' if s == '10k' else f'p339_nope1_lie_{s}'
R = [('lie r4', r4), ('nope1_path', f'p339_nope1_path_{s}')]
for m in ('smape', 'mse'):
    for L in (128, 256, 512):
        x = load(A, L, m); row = f'{s} ctx{c} {m} L{L} r2 {np.median(x):.3f}'
        for n, d in R:
            if not os.path.isdir(f'{E}/{d}_ls99_ctx{c}'): row += f' | {n} missing'; continue
            y = load(f'{E}/{d}_ls99_ctx{c}', L, m); I = [rng.integers(0, len(y), len(y)) for _ in range(300)]; bs = [np.median(x[i]) - np.median(y[i]) for i in I]
            row += f' | vs {n} {np.median(y):.3f} {np.median(x) - np.median(y):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]'
        print(row, flush=True)

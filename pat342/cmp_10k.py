"""PAT-342: nope1_lie 10k (none) vs released / p318_path final / p318_nope final / nope1_lie 4k, median sMAPE / MSE, paired bootstrap
95% CI (300 resamples, seed 0). usage: python pat342/cmp_10k.py CTX"""
import sys, os; sys.path.insert(0, '.')
import numpy as np
from pat318.agg import load
c = int(sys.argv[1]); E = 'pat318/eval'; rng = np.random.default_rng(0)
A = f'{E}/p342_n1lie10k_none_ls99_ctx{c}'
R = [('released', 'released'), ('path final', 'p318_path_s99'), ('nope final', 'p318_nope_s99'), ('lie 4k', 'p339_nope1_lie_s99_4k')]
for m in ('smape', 'mse'):
    for L in (128, 256, 512):
        x = load(A, L, m); row = f'ctx{c} {m} L{L} 10k {np.median(x):.3f}'
        for n, d in R:
            y = load(f'{E}/{d}_ls99_ctx{c}', L, m); I = [rng.integers(0, len(y), len(y)) for _ in range(300)]; bs = [np.median(x[i]) - np.median(y[i]) for i in I]
            row += f' | vs {n} {np.median(y):.3f} {np.median(x) - np.median(y):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]'
        print(row, flush=True)
